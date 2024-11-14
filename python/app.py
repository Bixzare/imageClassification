from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from models import Orchestrator  # Assuming your model is here
from config import Args  # Assuming Args is in your config file

# FastAPI setup
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.) or specifiy
    allow_headers=["*"],  # Allow all headers or specify
)

# Initialize model
args = Args(wandb_entity="votre_nom_entité")  # Adjust accordingly
model = Orchestrator(args)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

# Image preprocessing function
def preprocess_image(image_data: bytes,process: str):

    match process:
        case "Mosquito":
            pil_image = Image.open(BytesIO(image_data)).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize image to 224x224
                transforms.ToTensor(),
            ])
            return transform(pil_image).unsqueeze(0)
        
        case 'Pneumonia':
            # tbd
            return 0
        case 'Placeholder1':
            # tbd
            return 0

   

# Prediction function with post processing
def predict_image(tensor,process: str):
    match process:
        case "Mosquito":
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)

                    # # Return class based on predicted class
                    # if predicted_class.item() == 0:
                    #     return "AE"
                    # elif predicted_class.item() == 1:
                    #     return "AL"
                    # elif predicted_class.item() == 2:
                    #     return "JA"
                    # elif predicted_class.item() == 3:
                    #     return "KO"
                    # else:
                    #     raise NotImplementedError()
                     # Define the classes for easier mapping
                    # Create a list of class-probability pairs, with class names included
                    classes = ["AE", "AL", "JA", "KO"]
                    class_probabilities = [
                        {"class": classes[i], "probability": round(probabilities[0][i].item() * 100, 1)} 
                        for i in range(probabilities.size(1))
                    ]

                    # Return the result with predicted class and class probabilities
                    result = {
                        "pred": classes[predicted_class.item()],
                        "classes": class_probabilities
                    }

                    return result


# FastAPI endpoint for prediction
@app.post("/predict")
async def predict_image_and_process(image: UploadFile = File(...), process: str = Form(...)):
    try:
        # Read the image file into memory
        image_data = await image.read()

        # Preprocess the image
        tensor = preprocess_image(image_data,process)

        # Run prediction on the image
        result = predict_image(tensor,process)

        # Optionally, you can also log or process the `process` string
        image_name = image.filename  # Get the file name of the uploaded image
        print(f"Received process: {process}, Image: {image_name}")

        # Return the result as a JSON response
        return JSONResponse(content={"process": process, "imageName": image_name, "result": result})
    
    except Exception as e:
            
        print(f"Error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host = "127.0.0.1", port = 8000, reload = True)



# python -m uvicorn python.app:app --reload (in root)
