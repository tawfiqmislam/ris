import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import torch
import clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Load FAISS index + ID map
index = faiss.read_index("index.faiss")
ids = np.load("ids.npy")


@app.get("/")
async def root():
    return {"message": "Welcome to the CLIP-based image search API!"}


@app.post("/search-vector")
async def search_vector(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.encode_image(image_tensor).cpu().numpy()
        faiss.normalize_L2(vector)

    D, I = index.search(vector, 10)

    matched_ids = ids[I[0]].tolist()
    scores = D[0].tolist()

    results = []
    for i, sim in zip(matched_ids, scores):
        if sim >= 0.7:  # only keep reasonably similar matches
            results.append({"id": i, "similarity": round(sim, 4)})

    if not results:
        return JSONResponse(content={"results": [], "message": "No similar products found."})

    return {"results": results}



@app.post("/get-vector")
async def get_vector(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    # Convert to plain Python list
    vector = image_features[0].cpu().numpy().tolist()
    return JSONResponse(content={"vector": vector})



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

# @app.get("/ping")
# async def ping():
#     return {"message": "pong"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
