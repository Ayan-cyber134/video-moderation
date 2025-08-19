const express = require('express');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const blazeface = require('@tensorflow-models/blazeface');
const natural = require('natural');
const Jimp = require('jimp');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use('/uploads', express.static('uploads'));
app.use('/frames', express.static('frames'));

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB limit
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /mp4|avi|mov|mkv|webm/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = file.mimetype.startsWith('video/');

    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only video files are allowed'));
    }
  }
});

// Text moderation using NLP (hard oof)
class TextModerator {
  constructor() {
    this.tokenizer = new natural.WordTokenizer();
    this.stemmer = natural.PorterStemmer;
    
    // Comprehensive banned words list
    this.bannedWords = new Set([
      // Violence
      'kill', 'murder', 'attack', 'harm', 'violence', 'fight', 'war', 'weapon',
      'gun', 'knife', 'bomb', 'explode', 'shoot', 'stab', 'assault',
      
      // Hate speech
      'hate', 'racist', 'sexist', 'nazi', 'supremacist', 'bigot', 'discriminate',
      'slur', 'offensive', 'derogatory',
      
      // Explicit content
      'porn', 'xxx', 'nude', 'naked', 'explicit', 'adult', 'sex', 'sexual',
      'erotic', 'orgy', 'fetish', 'bdsm',
      
      // Drugs and illegal activities
      'drug', 'cocaine', 'heroin', 'marijuana', 'weed', 'opioid', 'overdose',
      'illegal', 'crime', 'theft', 'rob', 'steal', 'fraud', 'scam',
      
      // Self harm
      'suicide', 'selfharm', 'cutting', 'depression', 'anxiety', 'mental',
      
      // Harassment
      'bully', 'harass', 'stalk', 'threat', 'intimidate', 'blackmail'
    ]);

    this.bannedPatterns = [
      /http[s]?:\/\/[^\s]+/gi,
      /[0-9]{3}-[0-9]{2}-[0-9]{4}/g,
      /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi,
      /\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b/g, // Credit card patterns
      /\b\d{16}\b/g
    ];
  }

  moderateText(text) {
    if (!text || text.trim().length === 0) {
      return {
        hasIssues: false,
        bannedWords: [],
        hasSuspiciousPatterns: false,
        toxicityScore: 0,
        severity: 'NONE'
      };
    }

    const tokens = this.tokenizer.tokenize(text.toLowerCase());
    const stemmedTokens = tokens.map(token => this.stemmer.stem(token));
    
    const foundBannedWords = [];
    let hasSuspiciousPattern = false;

    // Check for banned words
    stemmedTokens.forEach((token, index) => {
      if (this.bannedWords.has(token)) {
        foundBannedWords.push({
          word: tokens[index],
          position: index,
          stemmed: token
        });
      }
    });

    // Check for suspicious patterns
    for (const pattern of this.bannedPatterns) {
      if (pattern.test(text)) {
        hasSuspiciousPattern = true;
        break;
      }
    }

    const toxicityScore = this.calculateToxicityScore(text);
    
    return {
      hasIssues: foundBannedWords.length > 0 || hasSuspiciousPattern || toxicityScore > 0.6,
      bannedWords: foundBannedWords,
      hasSuspiciousPatterns: hasSuspiciousPattern,
      toxicityScore: toxicityScore,
      severity: this.calculateSeverity(foundBannedWords.length, toxicityScore)
    };
  }

  calculateToxicityScore(text) {
    const negativeIndicators = [
      'hate', 'stupid', 'idiot', 'moron', 'retard', 'kill', 'die', 'worthless',
      'useless', 'disgusting', 'filthy', 'trash', 'ugly', 'fat', 'stupid',
      'dumb', 'loser', 'failure', 'pathetic'
    ];

    const tokens = this.tokenizer.tokenize(text.toLowerCase());
    let negativeCount = 0;

    tokens.forEach(token => {
      if (negativeIndicators.includes(token)) {
        negativeCount++;
      }
    });

    return Math.min(negativeCount / Math.max(tokens.length, 1) * 3, 1.0);
  }

  calculateSeverity(bannedWordCount, toxicityScore) {
    if (bannedWordCount > 3 || toxicityScore > 0.8) return 'HIGH';
    if (bannedWordCount > 1 || toxicityScore > 0.5) return 'MEDIUM';
    if (bannedWordCount > 0 || toxicityScore > 0.3) return 'LOW';
    return 'NONE';
  }
}

// Image moderation using TensorFlow.js (Very Professional lol)
class ImageModerator {
  constructor() {
    this.cocoModel = null;
    this.faceModel = null;
    this.initializeModels();
  }

  async initializeModels() {
    try {
      console.log('Loading COCO-SSD model...');
      this.cocoModel = await cocoSsd.load();
      console.log('COCO-SSD model loaded');
      
      console.log('Loading BlazeFace model...');
      this.faceModel = await blazeface.load();
      console.log('BlazeFace model loaded');
    } catch (error) {
      console.error('Failed to load models:', error);
    }
  }

  async moderateImage(imagePath) {
    if (!this.cocoModel || !this.faceModel) {
      throw new Error('Models not loaded yet');
    }

    try {
      // Read and preprocess image (always read and learn)
      const imageBuffer = fs.readFileSync(imagePath);
      const imageTensor = tf.node.decodeImage(imageBuffer);
      
      // Object detection
      const objects = await this.cocoModel.detect(imageTensor);
      
      // Face detection
      const faces = await this.faceModel.estimateFaces(imageTensor, false);
      
      // Dispose tensor
      imageTensor.dispose();

      // Analyze results
      const detectedObjects = objects.map(obj => ({
        class: obj.class,
        score: obj.score,
        bbox: obj.bbox
      }));

      const detectedFaces = faces.map(face => ({
        probability: face.probability ? face.probability[0] : null,
        landmarks: face.landmarks
      }));

      const issues = this.analyzeDetections(detectedObjects, detectedFaces);

      return {
        hasIssues: issues.length > 0,
        detectedObjects,
        detectedFaces: detectedFaces.length,
        issues,
        severity: this.calculateImageSeverity(issues)
      };

    } catch (error) {
      console.error('Image moderation error:', error);
      return {
        hasIssues: false,
        detectedObjects: [],
        detectedFaces: 0,
        issues: ['Analysis failed'],
        severity: 'UNKNOWN'
      };
    }
  }

  analyzeDetections(objects, faces) {
    const issues = [];
    
    // Check for inappropriate objects
    const inappropriateObjects = ['person', 'cell phone', 'book', 'laptop', 'tv'];
    const highRiskObjects = ['knife', 'gun', 'pistol', 'rifle', 'weapon'];
    
    objects.forEach(obj => {
      if (highRiskObjects.includes(obj.class.toLowerCase()) && obj.score > 0.5) {
        issues.push(`High-risk object detected: ${obj.class} (confidence: ${obj.score.toFixed(2)})`);
      }
      
      if (inappropriateObjects.includes(obj.class.toLowerCase()) && obj.score > 0.7) {
        issues.push(`Potentially inappropriate object: ${obj.class}`);
      }
    });

    // Check for faces (could indicate personal content)
    if (faces.length > 0) {
      issues.push(`Detected ${faces.length} face(s) - potential privacy concern`);
    }

    // Check for too many objects (could be spam)
    if (objects.length > 10) {
      issues.push(`High object count (${objects.length}) - potential spam`);
    }

    return issues;
  }

  calculateImageSeverity(issues) {
    const highRiskKeywords = ['High-risk', 'weapon', 'gun', 'knife'];
    const hasHighRisk = issues.some(issue => 
      highRiskKeywords.some(keyword => issue.includes(keyword))
    );

    if (hasHighRisk) return 'HIGH';
    if (issues.length > 2) return 'MEDIUM';
    if (issues.length > 0) return 'LOW';
    return 'NONE';
  }
}

// Video processing utilities
class VideoProcessor {
  static extractFrames(videoPath, outputDir, interval = '00:00:02') {
    return new Promise((resolve, reject) => {
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }

      // Get video duration first
      ffmpeg.ffprobe(videoPath, (err, metadata) => {
        if (err) return reject(err);
        
        const duration = metadata.format.duration;
        const frameCount = Math.min(Math.floor(duration / 2), 10); // Max 10 frames
        
        const timestamps = [];
        for (let i = 1; i <= frameCount; i++) {
          const timestamp = Math.floor((duration / frameCount) * i);
          timestamps.push(timestamp);
        }

        ffmpeg(videoPath)
          .on('end', () => resolve(outputDir))
          .on('error', (err) => reject(err))
          .screenshots({
            timestamps: timestamps,
            filename: 'frame-%i.png',
            folder: outputDir,
            size: '320x240' // Smaller size for faster processing (lol)
          });
      });
    });
  }

  static extractAudio(videoPath, outputPath) {
    return new Promise((resolve, reject) => {
      ffmpeg(videoPath)
        .output(outputPath)
        .noVideo()
        .audioCodec('pcm_s16le')
        .on('end', () => resolve(outputPath))
        .on('error', (err) => reject(err))
        .run();
    });
  }
}

// Initialize moderators
const textModerator = new TextModerator();
const imageModerator = new ImageModerator();

// Routes
app.post('/api/moderate/video', upload.single('video'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file provided' });
    }

    const videoPath = req.file.path;
    const videoId = path.basename(videoPath).split('-')[0];
    const framesDir = path.join('frames', videoId);

    console.log(`Processing video: ${videoPath}`);
    
    // Extract frames from video
    console.log('Extracting frames...');
    await VideoProcessor.extractFrames(videoPath, framesDir);
    
    // Analyze each frame
    console.log('Analyzing frames...');
    const frameFiles = fs.readdirSync(framesDir)
      .filter(file => file.endsWith('.png'))
      .map(file => path.join(framesDir, file));

    const frameResults = [];
    for (const frameFile of frameFiles) {
      try {
        const result = await imageModerator.moderateImage(frameFile);
        frameResults.push({
          frame: path.basename(frameFile),
          result: result
        });
      } catch (error) {
        console.error(`Error analyzing frame ${frameFile}:`, error);
        frameResults.push({
          frame: path.basename(frameFile),
          error: error.message
        });
      }
    }

    // Simulate text analysis (would need speech-to-text in real implementation) (Add it urself as APIs are costly)
    const textAnalysis = textModerator.moderateText(
      "Sample text analysis - in real implementation, extract audio and convert to text"
    );

    // Calculate overall moderation result
    const overallResult = calculateOverallModeration(frameResults, textAnalysis);

    res.json({
      success: true,
      filename: req.file.originalname,
      videoId: videoId,
      framesAnalyzed: frameResults.length,
      frameResults: frameResults,
      textAnalysis: textAnalysis,
      overallResult: overallResult,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Video moderation error:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
});

app.post('/api/moderate/image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    const result = await imageModerator.moderateImage(req.file.path);
    
    // Clean up (need vacuum cleaner)
    cleanupFile(req.file.path);

    res.json({
      success: true,
      filename: req.file.originalname,
      moderation: result
    });

  } catch (error) {
    console.error('Image moderation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.post('/api/moderate/text', (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'No text provided' });
    }

    const moderationResult = textModerator.moderateText(text);

    res.json({
      success: true,
      text: text.substring(0, 500) + (text.length > 500 ? '...' : ''),
      moderation: moderationResult
    });

  } catch (error) {
    console.error('Text moderation error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

// Helper functions
function calculateOverallModeration(frameResults, textAnalysis) {
  let maxSeverity = 'NONE';
  const severityLevels = { 'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'UNKNOWN': 0 };

  // Check frame results
  frameResults.forEach(frame => {
    if (frame.result && severityLevels[frame.result.severity] > severityLevels[maxSeverity]) {
      maxSeverity = frame.result.severity;
    }
  });

  // Check text analysis
  if (severityLevels[textAnalysis.severity] > severityLevels[maxSeverity]) {
    maxSeverity = textAnalysis.severity;
  }

  const hasIssues = maxSeverity !== 'NONE';
  const issueCount = frameResults.filter(f => f.result && f.result.hasIssues).length + 
                    (textAnalysis.hasIssues ? 1 : 0);

  return {
    status: hasIssues ? 'REVIEW' : 'PASS',
    severity: maxSeverity,
    totalIssues: issueCount,
    requiresHumanReview: hasIssues
  };
}

function cleanupFile(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (error) {
    console.warn('Could not clean up file:', filePath);
  }
}

// Health check endpoint (need doctor)
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    modelsLoaded: !!imageModerator.cocoModel && !!imageModerator.faceModel
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// 404 handler (Always for u)
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Video Moderation API running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});

// Handle graceful shutdown (heheh)
process.on('SIGINT', () => {
  console.log('Shutting down gracefully...');
  process.exit(0);
});
