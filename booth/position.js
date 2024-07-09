const Tesseract = require('tesseract.js');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');
const sharp = require('sharp');

function parseImage(imagePath) {
  const processedImagePath = path.join(__dirname, 'input.png');
  const processWidth = 4000;
  const outputWidth = 1920;
  const resizedImagePath = path.join(__dirname, 'input-resized.png');
  sharp(imagePath)
    .resize({ width: outputWidth })
    .toFile(resizedImagePath, (err, info) => {
      if (err) {
        console.error('Error processing image:', err);
        return;
      }
      sharp(imagePath)
        .resize({ width: processWidth })
        .grayscale()
        .threshold(128) // Binarize the image
        .normalize() // Normalize contrast
        .toFile(processedImagePath, (err, info) => {
          if (err) {
            console.error('Error processing image:', err);
            return;
          }
          parseCanvas(processedImagePath, resizedImagePath, { processWidth, outputWidth });
        });
    })
  
}

function parseCanvas(imagePath, originalImagePath, { processWidth, outputWidth }) {
  loadImage(originalImagePath).then((originalImg) => {
  loadImage(imagePath).then((image) => {
    parseText(image, originalImg, { processWidth, outputWidth })
  })})
}

function parseText(image, originalImage, { processWidth, outputWidth }) {
  const canvas = createCanvas(image.width, image.height);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, image.width, image.height);
  console.log('image size', image.width, image.height);

  const canvasOuput = createCanvas(originalImage.width, originalImage.height);
  const ctxOuput = canvasOuput.getContext('2d');
  ctxOuput.drawImage(originalImage, 0, 0, originalImage.width, originalImage.height);

  const config = {
    tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', // Optional: Limit character set
    tessedit_pageseg_mode: Tesseract.PSM.SINGLE_BLOCK,
    tessedit_ocr_engine_mode: Tesseract.OEM.DEFAULT, // Higher accuracy mode
    user_defined_dpi: 300, // Optional: Define DPI
  };
  Tesseract.recognize(
    canvas.toDataURL(),
    'eng',
    {
      config
    }
    // { logger: m => console.log(m) }
  )
    .then(({ data: { text, words } }) => {
      console.log('Recognized Text:\n', text);
      console.log('\nText Positions:');

      const filteredWords = words.map((word) => {
        // replace all non alphanumeric characters with a space
        const originalText = word.text;
        word.text = word.text.replace(/[^a-zA-Z0-9]/g,'').trim()
        if (originalText.length !== word.text.length) {
          console.log('original text:', originalText, word.text);
        }
        return word
      }).filter(word => {
        if (word.text.length < 4) {
          return false;
        }
        if (word.text.length > 4) {
          return false;
        }

        return Number.parseInt(word.text[0], 10) >= 0;
      })

      for (const word of filteredWords) {
        const { text, bbox } = word;
        const scaledBBox = {
          x0: bbox.x0 * outputWidth / processWidth,
          y0: bbox.y0 * outputWidth / processWidth,
          x1: bbox.x1 * outputWidth / processWidth,
          y1: bbox.y1 * outputWidth / processWidth,
        };

        console.log(
          `Text: ${text}`,
          { scaledBBox, bbox }
        );

        // draw it in red
        ctxOuput.strokeStyle = 'red';
        ctxOuput.lineWidth = 2;
        // ctxOuput.strokeRect(bbox.x0, bbox.y0, bbox.x1 - bbox.x0, bbox.y1 - bbox.y0);
        ctxOuput.strokeRect(scaledBBox.x0, scaledBBox.y0, scaledBBox.x1 - scaledBBox.x0, scaledBBox.y1 - scaledBBox.y0);
      }
      // write canvas to output.png
      const out = fs.createWriteStream('./output.png');
      const stream = canvasOuput.createPNGStream();
      stream.pipe(out);
      out.on('finish', () => console.log('The file was created.'));
      console.log('Done.');
    })
    .catch((err) => {
      console.error(err);
    });
}

parseImage(path.resolve(__dirname, 'seacare-layout-new.jpg'));
