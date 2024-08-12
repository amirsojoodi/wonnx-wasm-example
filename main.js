// import init, {Input, main, Session} from '@webonnx/wonnx-wasm';
import init, {Input, main, Session} from './node_modules/@webonnx/wonnx-wasm/wonnx.js';

async function fetchBytes(url) {
  const reply = await fetch(url);
  const blob = await reply.arrayBuffer();
  const arr = new Uint8Array(blob);
  return arr;
}

async function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.src = url;
    image.onload = () => resolve(image);
    image.onerror = (err) => reject(err);
  });
}

async function run(imageUrl) {
  try {
    // Load model, labels file and WONNX
    const labels =
        fetch('./data/models/squeeze-labels.txt').then(r => r.text());
    const [modelBytes, initResult, labelsResult] = await Promise.all(
        [fetchBytes('./data/models/opt-squeeze.onnx'), init(), labels])
    console.log('Initialized', {modelBytes, initResult, Session, labelsResult});
    const squeezeWidth = 224;
    const squeezeHeight = 224;

    // Start inference session
    const session = await Session.fromBytes(modelBytes);

    // Parse labels
    const labelsList = labelsResult.split(/\n/g);
    console.log({labelsList});

    const imageElement = document.getElementById('image');

    // Create a canvas to capture video frames
    imageElement.src = imageUrl;

    const canvas = document.createElement('canvas');
    canvas.width = squeezeWidth;
    canvas.height = squeezeHeight;
    const context = canvas.getContext('2d', {willReadFrequently: true});

    let inferenceCount = 0;
    let inferenceTime = 0;

    // Captures a frame and produces inference
    async function inferImage() {
      try {
        const inputImage = await loadImage(imageUrl);

        context.drawImage(inputImage, 0, 0, canvas.width, canvas.height);

        // document.body.appendChild(canvas);
        // document.body.appendChild(canvas);
        const data = context.getImageData(0, 0, canvas.width, canvas.height);
        const image = new Float32Array(224 * 224 * 3);
        // Transform the image data in the format expected by SqueezeNet
        const planes = 3;          // SqueezeNet expects RGB
        const valuesPerPixel = 4;  // source data is RGBA
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        for (let plane = 0; plane < planes; plane++) {
          for (let y = 0; y < squeezeHeight; y++) {
            for (let x = 0; x < squeezeWidth; x++) {
              const v =
                  data.data[y * squeezeWidth * valuesPerPixel + x * valuesPerPixel + plane] /
                  255.0;
              image[plane * (squeezeWidth * squeezeHeight) + y * squeezeWidth + x] =
                  (v - mean[plane]) / std[plane];
            }
          }
        }
        // Start inference
        const input = new Input();
        input.insert('data', image);
        const start = performance.now();
        const result = await session.run(input);
        const duration = performance.now() - start;
        input.free();

        // Find the label with the highest probability
        const probs = result.get('squeezenet0_flatten0_reshape0');

        const numLabels = 5;
        let resultLabels = '';

        for (let r = 0; r < numLabels; r++) {
          let maxProb = -1;
          let maxIndex = -1;
          for (let index = 0; index < probs.length; index++) {
            const p = probs[index];
            if (p > maxProb) {
              maxProb = p;
              maxIndex = index;
            }
          }

          resultLabels += `${labelsList[maxIndex]} (${maxProb})\n`;
          probs[maxIndex] = 0;
        }

        document.getElementById('log').innerText = resultLabels;
        document.getElementById('perf').innerText =
            `Inference time: ${duration.toFixed(2)}ms`;
      } catch (e) {
        console.error(e, e.toString());
      }
    }
    await inferImage();

  } catch (e) {
    console.error(e, e.toString());
  }
}
const imageUrl = './data/images/input1.jpg';

run(imageUrl);