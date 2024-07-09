const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');
const fs = require('fs');
const { createCanvas, Image } = require('canvas');

async function detectBooths(imagePath) {
  // Load the image
  const img = new Image();
  img.src = fs.readFileSync(imagePath);

  // Create a canvas
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, img.width, img.height);

  // Load the COCO-SSD model
  const model = await cocoSsd.load();

  // Make predictions
  const predictions = await model.detect(canvas);
  console.log('Predictions:', predictions);
  // Filter predictions for booths (you might need to adjust this based on your use case)
  const booths = predictions.filter(prediction => prediction.class === 'booth');
  console.log('Found', booths.length, 'booths');
  // Draw rectangles around detected booths and log coordinates
  for (const booth of booths) {
    const { bbox, score } = booth;
    const [x, y, width, height] = bbox;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, width, height);
    console.log(`Booth detected at [x: ${x}, y: ${y}, width: ${width}, height: ${height}], confidence: ${score}`);
  }

  // Save the image with detected booths
  const out = fs.createWriteStream('./output.png');
  const stream = canvas.createPNGStream();
  stream.pipe(out);
  out.on('finish', () => console.log('The file was created.'));
}

// Detect booths in the image
// detectBooths('seacare-layout-new-hd.jpg');

detectBooths('istockphoto-1407814845-1024x1024.jpg');