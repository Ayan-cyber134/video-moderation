## 1. Usage Instructions

1. **Install dependencies:**
```bash
npm install
```

2. **Download AI models:**
```bash
npm run setup
```

3. **Start the server:**
```bash
npm run dev
```

4. **API Endpoints:**
   - `POST /api/moderate/video` - Upload and moderate video
   - `POST /api/moderate/image` - Upload and moderate image
   - `POST /api/moderate/text` - Moderate text content
   - `GET /health` - Health check

## 2. Example Usage with curl

```bash
# Moderate video
curl -X POST -F "video=@your-video.mp4" http://localhost:3000/api/moderate/video

# Moderate image
curl -X POST -F "image=@your-image.jpg" http://localhost:3000/api/moderate/image

# Moderate text
curl -X POST -H "Content-Type: application/json" -d '{"text":"your text here"}' http://localhost:3000/api/moderate/text
```
