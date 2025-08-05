
# Wahlinapp RIP

A brief description of what this project does and who it's for


## Installation

Install with pip using requirements.txt file

For project run
```bash
  python main.py
```

For build index
```bash
  create product_vectors.json, ids.npy, index.faiss file

  get product vector using 
  curl -X POST 'http://127.0.0.1:8001/get-vector' \  -F 'file=@image.png'

  product_vectors.json file contail
    [
      {
        "id": 1,
        "vector": [...]
      }
    ]
    type data

  python build_index.py
```

For Generate Vectors [ its generate product_vectors.json file data ]
```bash
  create product_image_map.json file and ensure product_vectors.json exist

  put images on images folder and maping image stucture in product_image_map.json
  its look like this
  {
    "1": [
        "thumbnail/2025-01-21-678fed310f2a8.webp",
        "2025-01-21-678fed3108fdf.webp",
        "2025-01-21-678fed310a8b0.webp",
        "2025-01-21-678fed310bf94.webp",
        "2025-01-21-678fed310d7cc.webp"
    ],
    ...
  }

  python generate_vectors.py

```