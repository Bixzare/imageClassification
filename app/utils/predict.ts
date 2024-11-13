import { getImageTensorFromPath } from "./imageHelper";
import { runModel } from "./modelHelper";

export async function mosquitoModel(path : string): Promise<[any,number]>{

    const imageTensor = await getImageTensorFromPath(path);

    const modelPath = './public/model.onnx'
    const [predictions, inferenceTime] = await runModel(imageTensor)
    
    return [predictions, inferenceTime];
}