import os
import json

# === Config ===
OUTPUT_JSON_PATH = "custom_image_map.json"


def read_images_with_folders_simple(root_folder):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    output_data = {}

    for root, dirs, files in os.walk(root_folder):
        folder_name = os.path.basename(root)
        relative_path = os.path.relpath(root, root_folder)
        
        image_paths = []
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(root, file)

                image_paths.append(f"{root_folder}/{relative_path}/{file}")

        if image_paths:
            output_data[relative_path] = image_paths
    
    return output_data


def main():
    root_directory = "images/custom"
    results = read_images_with_folders_simple(root_directory)

    # Save to JSON
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved custom image path to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()