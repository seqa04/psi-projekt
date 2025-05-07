import zipfile
import os
import json
import random
import shutil
from collections import defaultdict

zip_path = "psi_data.zip"
output_dir = "dataset"
train_ratio = 0.8
val_ratio = 0.1
random.seed(42)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

annotation_file = None
image_dir = None

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".json"):
            annotation_file = os.path.join(root, file)
        elif file.lower().endswith((".jpg")):
            image_dir = root
        


assert annotation_file and image_dir, "Nie znaleziono zdjęć lub adnotacji!"

with open(annotation_file) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

image_id_to_categories = defaultdict(set)
for ann in annotations:
    image_id_to_categories[ann["image_id"]].add(ann["category_id"])

image_category_pairs = [(img, list(image_id_to_categories[img["id"]])) for img in images]
random.shuffle(image_category_pairs)

split_sizes = {
    "train": int(train_ratio * len(images)),
    "val": int(val_ratio * len(images)),
    "test": len(images) - int(train_ratio * len(images)) - int(val_ratio * len(images))
}

splits = {"train": [], "val": [], "test": []}
category_counts = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}


for img, cats in image_category_pairs:
    # Znajdź split, gdzie te kategorie są najmniej obecne
    best_split = None
    min_sum = float('inf')
    for split in ["train", "val", "test"]:
        if len(splits[split]) >= split_sizes[split]:
            continue
        score = sum(category_counts[split][cat] for cat in cats)
        if score < min_sum:
            min_sum = score
            best_split = split
    if best_split:
        splits[best_split].append(img)
        for cat in cats:
            category_counts[best_split][cat] += 1

# === 6. Filtrowanie adnotacji i zapisywanie wyników ===
def get_annotations_for(images_subset):
    ids = {img["id"] for img in images_subset}
    return [ann for ann in annotations if ann["image_id"] in ids]

def save_split(name, image_subset):
    path = os.path.join(output_dir, name)
    os.makedirs(path, exist_ok=True)

    # kopiuj obrazy
    for img in image_subset:
        src = os.path.join(image_dir, img["file_name"])
        dst = os.path.join(path, img["file_name"])
        if os.path.exists(src):
            shutil.copy(src, dst)

    # zapisz JSON
    split_json = {
        "images": image_subset,
        "annotations": get_annotations_for(image_subset),
        "categories": categories
    }
    with open(os.path.join(path, f"{name}.json"), "w") as f:
        json.dump(split_json, f)

# Zapisz wszystkie
for split in ["train", "val", "test"]:
    save_split(split, splits[split])