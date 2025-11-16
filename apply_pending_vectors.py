import json

MAIN_JSON_FILE = "product_vectors.json"
TMP_JSON_FILE = "product_vectors_pending.json"

def main():
    try:
        with open(TMP_JSON_FILE, "r") as f:
            tmp_vectors = json.load(f)
    except Exception as e:
        return
    
    if not tmp_vectors:
        return

    try:
        with open(MAIN_JSON_FILE, "r") as f:
            product_vectors = json.load(f)
    except Exception as e:
        product_vectors = []

    productDick = {item["id"]: item for item in product_vectors}

    for item in tmp_vectors:
        productDick[item["id"]] = item

    productList = list(productDick.values())

    with open(MAIN_JSON_FILE, "w") as f:
        json.dump(productList, f, indent=2)

    with open(TMP_JSON_FILE, "w") as f:
        json.dump([], f, indent=2)


if __name__ == "__main__":
    main()