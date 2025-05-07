import zipfile
import os
import json
import random
import shutil

# === KONFIGURACJA ===
zip_path = "psi_data.zip"
output_dir = "dataset"
train_ratio = 0.8
val_ratio = 0.1
random.seed(42)

# === 1. Rozpakuj plik zip ===
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

# Znajdź plik z adnotacjami
annotation_file = None
image_dir = None

for root, dirs, files in os.walk(output_dir):
    for file in files:
        if file.endswith(".json"):
            annotation_file = os.path.join(root, file)
        elif file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_dir = root

assert annotation_file and image_dir, "Nie znaleziono zdjęć lub adnotacji!"

# === 2. Wczytaj dane COCO ===
with open(annotation_file) as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# === 3. Podziel dane ===
random.shuffle(images)
n = len(images)
train_images = images[:int(train_ratio*n)]
val_images = images[int(train_ratio*n):int((train_ratio+val_ratio)*n)]
test_images = images[int((train_ratio+val_ratio)*n):]

def filter_annotations(image_subset):
    ids = {img["id"] for img in image_subset}
    return [ann for ann in annotations if ann["image_id"] in ids]

# === 4. Funkcja do zapisu danych ===
def save_split(split_name, image_subset):
    split_path = os.path.join(output_dir, split_name)
    os.makedirs(split_path, exist_ok=True)

    # kopiuj obrazy
    for img in image_subset:
        img_path = os.path.join(image_dir, img["file_name"])
        shutil.copy(img_path, os.path.join(split_path, img["file_name"]))

    # zapisz JSON
    split_json = {
        "images": image_subset,
        "annotations": filter_annotations(image_subset),
        "categories": categories
    }

    with open(os.path.join(split_path, f"{split_name}.json"), 'w') as f:
        json.dump(split_json, f)

save_split("train", train_images)
save_split("val", val_images)
save_split("test", test_images)