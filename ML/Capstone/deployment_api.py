
"""
Makeup Detection API Service
FastAPI application for serving the makeup detection model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json

# Initialize FastAPI app
app = FastAPI(title="Makeup Detection API", version="1.0.0")

# Load model configuration
with open('model_config.json', 'r') as f:
    config = json.load(f)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load('makeup_detection_model.pt', map_location=device)
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['normalization']['mean'],
                        std=config['normalization']['std'])
])

@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "healthy", "model": "Makeup Detection ResNet18"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict makeup attributes from an uploaded image

    Args:
        file: Image file (JPG, PNG)

    Returns:
        JSON with predictions for each attribute
    """
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Transform image
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output).cpu().numpy()[0]
            predictions = (probabilities > config['threshold']).astype(int)

        # Format response
        results = {}
        for i, attr_name in enumerate(config['attributes']):
            results[attr_name] = {
                'prediction': 'Yes' if predictions[i] == 1 else 'No',
                'confidence': float(probabilities[i])
            }

        return JSONResponse(content={
            'success': True,
            'predictions': results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    """Get model information"""
    return {
        'model': config['model_name'],
        'attributes': config['attributes'],
        'input_size': config['input_size'],
        'device': str(device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
