import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import faiss

# === Initialize FastAPI ===
app = FastAPI()

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load MobileNet ===
from torchvision.models import MobileNet_V2_Weights
mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
mobilenet.classifier = nn.Identity()  # remove final classification layer
mobilenet.eval().to(device)

# === Preprocessing (same as MobileNet expects) ===
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# === Load FAISS index + ID map ===
index = faiss.read_index("index.faiss")
ids = np.load("ids.npy")


@app.get("/")
async def root():
    return {"message": "Welcome to the CLIP-based image search API!"}

# === Extract feature vector ===
def extract_vector(image: Image.Image):
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = mobilenet(image_tensor).cpu().numpy()
    faiss.normalize_L2(vector)
    return vector


@app.post("/get-vector")
async def get_vector(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    vector = extract_vector(image)
    return JSONResponse(content={"vector": vector[0].tolist()})


@app.post("/search-vector")
async def search_vector(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    vector = extract_vector(image)

    D, I = index.search(vector, 10)
    matched_ids = ids[I[0]].tolist()
    scores = D[0].tolist()

    results = []
    for i, sim in zip(matched_ids, scores):
        if sim >= 0.5:
            results.append({"id": int(i), "similarity": round(float(sim), 4)})

    if not results:
        return JSONResponse(content={"results": [], "message": "No similar products found."})

    return {"results": results}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
