const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const blazeface = require('@tensorflow-models/blazeface');

async function downloadModels() {
  console.log('Downloading COCO-SSD model...');
  await cocoSsd.load();
  console.log('COCO-SSD model downloaded');
  
  console.log('Downloading BlazeFace model...');
  await blazeface.load();
  console.log('BlazeFace model downloaded');
  
  console.log('All models downloaded successfully!');
}

downloadModels().catch(console.error);
