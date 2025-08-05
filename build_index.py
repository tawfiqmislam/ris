# import faiss
# import numpy as np
# import json

# # Load vectors from JSON
# with open("product_vectors.json", "r") as f:
#     data = json.load(f)  # Format: [{ "id": 101, "vector": [...] }, ...]

# vectors = np.array([item["vector"] for item in data]).astype('float32')
# ids = np.array([item["id"] for item in data])

# # Create FAISS index
# index = faiss.IndexFlatIP(vectors.shape[1])  # Use cosine similarity
# faiss.normalize_L2(vectors)  # Normalize for cosine similarity

# index.add(vectors)  # Add vectors to index

# # Save index and ID map
# faiss.write_index(index, "index.faiss")
# np.save("ids.npy", ids)

# print(f"Indexed {len(ids)} products.")

# import faiss
# import numpy as np
# import json

# with open("product_vectors.json", "r") as f:
#     data = json.load(f)

# # vectors = np.array([item["vector"] for item in data]).astype("float32")
# # ids = np.array([item["id"] for item in data])
# valid_data = [item for item in data if isinstance(item.get("vector"), list) and len(item["vector"]) == 512]
# vectors = np.array([item["vector"] for item in valid_data]).astype("float32")
# ids = np.array([item["id"] for item in valid_data])


# # Normalize
# faiss.normalize_L2(vectors)

# # Build and save index
# # Use IndexIVFFlat for huge datasets or IndexFlatIP	Replace IndexFlatIP with IndexIVFFlat and train it
# index = faiss.IndexFlatIP(vectors.shape[1])
# index.add(vectors)
# faiss.write_index(index, "index.faiss")
# np.save("ids.npy", ids)

# print(f"Indexed {len(ids)} products.")

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
    and len(item["vector"]) == 512
    and isinstance(item.get("id"), int)
]

if not valid_data:
    raise ValueError("No valid vectors found in the JSON file.")

# Prepare arrays
vectors = np.array([item["vector"] for item in valid_data], dtype="float32")
ids = np.array([item["id"] for item in valid_data], dtype="int32")

# Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)

# Build FAISS index
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

# Save index and ID map
faiss.write_index(index, INDEX_FILE)
np.save(ID_MAP_FILE, ids)

print(f"✅ Indexed {len(ids)} products and saved to '{INDEX_FILE}' and '{ID_MAP_FILE}'.")
