/************ Create session and return results here */
// https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html
// https://fritz.ai/building-an-image-recognition-app-using-onnx-js/
// https://medium.com/@nico.axtmann95/scalable-image-classification-with-onnx-js-and-aws-lambda-ab3d7bd1723



























// import * as ort from 'onnxruntime-web';
// import _ from 'lodash';
// import { imagenetClasses } from '../data/imagenet';

// export async function runModel(preprocessedData: any): Promise<[any, number]> {
//   // Create session and set options. See the docs here for more options: 
//   //https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html#graphOptimizationLevel
//   try {
//     // Adjust the path based on where your model is located
//     const modelPath = './public/mode.onnx'; // or wherever your model is located

//     const session = await ort.InferenceSession.create(modelPath, {
//       executionProviders: ['webgl'], // or ['wasm'] based on your environment
//       graphOptimizationLevel: 'all',
//     });

//     console.log('Inference session created');
//     return [];
//   } catch (error) {
//     console.error('Error creating inference session:', error);
//   }
// }

// async function runInference(session: ort.InferenceSession, preprocessedData: any): Promise<[any, number]> {
//     // Get start time to calculate inference time.
//     const start = new Date();
//     // create feeds with the input name from model export and the preprocessed data.
//     const feeds: Record<string, ort.Tensor> = {};
//     feeds[session.inputNames[0]] = preprocessedData;
//     // Run the session inference.
//     const outputData = await session.run(feeds);
//     // Get the end time to calculate inference time.
//     const end = new Date();
//     // Convert to seconds.
//     const inferenceTime = (end.getTime() - start.getTime())/1000;
//     // Get output results with the output name from the model export.
//     const output = outputData[session.outputNames[0]];
//     //Get the softmax of the output data. The softmax transforms values to be between 0 and 1
//     var outputSoftmax = softmax(Array.prototype.slice.call(output.data));
//     //Get the top 5 results.
//     var results = imagenetClassesTopK(outputSoftmax, 4);
//     console.log('results: ', results);
//     return [results, inferenceTime];
//   }
  

// // const session = await ort.InferenceSession
// //                           .create('./_next/static/chunks/pages/squeezenet1_1.onnx', 
// //                           { executionProviders: ['webgl'], graphOptimizationLevel: 'all' });
// //   console.log('Inference session created');
// //   // Run inference and get results.
// //   var [results, inferenceTime] =  await runInference(session, preprocessedData);
// //   return [results, inferenceTime];