import os
import json
import requests
import numpy as np
from tqdm import tqdm

# === Config ===
BASE_IMAGE_DIR = "images"
VECTOR_API_URL = "http://127.0.0.1:8001/get-vector"
INPUT_JSON_PATH = "product_image_map.json"
OUTPUT_JSON_PATH = "product_vectors.json"

def get_vector(image_path):
    with open(image_path, "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(VECTOR_API_URL, files=files)
            response.raise_for_status()
            return response.json()["vector"]
        except Exception as e:
            print(f"❌ Failed for {image_path}: {e}")
            return None

def main():
    with open(INPUT_JSON_PATH, "r") as f:
        product_images = json.load(f)

    output_data = []

    print("🚀 Generating vectors...")
    for product_id, image_paths in tqdm(product_images.items()):
        vectors = []

        for rel_path in image_paths:
            full_path = rel_path if os.path.isabs(rel_path) else os.path.join(BASE_IMAGE_DIR, rel_path)
            if os.path.exists(full_path):
                vector = get_vector(full_path)
                if vector:
                    vectors.append(vector)
            else:
                print(f"⚠️ Image not found: {full_path}")

        if vectors:
            avg_vector = np.mean(np.array(vectors), axis=0).tolist()
            output_data.append({
                "id": int(product_id),
                "vector": avg_vector
            })

    # Save to JSON
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Saved vectors to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()
