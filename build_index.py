import faiss
import numpy as np
import json
import os

VECTOR_FILE = "product_vectors.json"
INDEX_FILE = "index.faiss"
ID_MAP_FILE = "ids.npy"

# Check file existence
if not os.path.exists(VECTOR_FILE):
    raise FileNotFoundError(f"{VECTOR_FILE} not found!")

# Load data
with open(VECTOR_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter and validate
valid_data = [
    item for item in data
    if isinstance(item.get("vector"), list)
    and len(item["vector"]) == 1280   # ✅ MobileNetV2 output size
    and isinstance(item.get("id"), int)
]

if not valid_data:
    raise ValueError("No valid 1280-dim vectors found in the JSON file. Did you rebuild with MobileNet?")

# Prepare arrays
vectors = np.array([item["vector"] for item in valid_data], dtype="float32")
ids = np.array([item["id"] for item in valid_data], dtype="int32")

# Normalize for cosine similarity
faiss.normalize_L2(vectors)

# Build and save index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, INDEX_FILE)
np.save(ID_MAP_FILE, ids)

print(f"✅ Indexed {len(ids)} products (dim={vectors.shape[1]}) and saved to '{INDEX_FILE}' and '{ID_MAP_FILE}'.")
