import zipfile
import os
import json
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

zip_path = "dataset.zip"
output_dir = "dataset"
test_ratio = 0.8
val_ratio = 0.1
random.seed(42)

def extract_zip(zip_path, output_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)


def find_data_paths(output_dir):
    annotation_file = None
    image_dir = None
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".json"):
                annotation_file = os.path.join(root, file)
            elif file.lower().endswith((".jpg")):
                image_dir = root
            
    assert annotation_file and image_dir, "Photos or Annotations not found"
    return annotation_file, image_dir

def load_coco_data(annotation_file):
    with open(annotation_file) as f:
        coco = json.load(f)
    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    return images, annotations, categories

def stratified_split(images, annotations, test_ratio, val_ratio):
    image_ids = np.array([img["id"] for img in images])
    zeros = np.zeros_like(image_ids)

    image_to_categories = np.stack((image_ids, zeros), axis=1)

    imageid_to_cat = {}
    for ann in annotations:
        if ann["image_id"] not in imageid_to_cat:
            imageid_to_cat[ann["image_id"]] = ann["category_id"]

    for i, row in enumerate(image_to_categories):
        img_id = row[0]
        if img_id in imageid_to_cat:
            image_to_categories[i, 1] = imageid_to_cat[img_id]

    X = image_to_categories[:, 0]
    y = image_to_categories[:, 1] 

    X_train, X_temp, _, y_temp = train_test_split(
        X, y, test_size=test_ratio+val_ratio, stratify=y, random_state=42
    )

    train_val_ratio = test_ratio/(test_ratio+val_ratio)
    X_val, X_test, _, _ = train_test_split(
        X_temp, y_temp, test_size=train_val_ratio, stratify=y_temp, random_state=42
    )

    splits = {"train":X_train,"test":X_test,"val":X_val}
    return splits


def save_split(name, image_ids, all_images, all_annotations, all_categories, image_dir, output_dir):
    path = os.path.join(output_dir, name)
    os.makedirs(path, exist_ok=True)

    id_to_image = {img["id"]: img for img in all_images}

    selected_images = [id_to_image[img_id] for img_id in image_ids if img_id in id_to_image]

    for img in selected_images:
        filename = img["file_name"]
        src = os.path.join(image_dir, filename)
        dst = os.path.join(path, filename)
        if os.path.exists(src):
            shutil.copy(src, dst)

    selected_image_ids = {img["id"] for img in selected_images}
    selected_annotations = [ann for ann in all_annotations if ann["image_id"] in selected_image_ids]

    split_json = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": all_categories
    }

    with open(os.path.join(path, f"{name}.json"), "w") as f:
        json.dump(split_json, f)

def cleanup(image_dir):
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
        print(f"Usunięto folder z oryginalnymi obrazami: {image_dir}")


def report_split(split_name, image_ids, annotations):
    ann_count = sum(1 for ann in annotations if ann["image_id"] in image_ids)
    cat_counter = Counter(
        ann["category_id"] for ann in annotations if ann["image_id"] in image_ids
    )
    sorted_counts = dict(sorted(cat_counter.items(), key=lambda x: x[0]))

    annotated_image_ids = set(ann["image_id"] for ann in annotations)
    image_ids_set = set(image_ids)
    unannotated_image_ids = image_ids_set - annotated_image_ids

    print(f"\nRaport dla zbioru: {split_name.upper()}")
    print(f" - Liczba obrazów: {len(image_ids)}")
    print(f" - Liczba adnotacji: {ann_count}")
    print(f" - Obrazów bez kategorii (sam background): {len(unannotated_image_ids)}")
    print(f" - Kategorie: {sorted_counts}")

def preprocess_dataset(zip_path, output_dir, test_ratio=0.1, val_ratio=0.1):
    extract_zip(zip_path, output_dir)
    annotation_file, image_dir = find_data_paths(output_dir)
    images, annotations, categories = load_coco_data(annotation_file)

    splits = stratified_split(images, annotations, test_ratio, val_ratio)

    for split in ["train", "val", "test"]:
        report_split(split, splits[split], annotations)
        save_split(split, splits[split], images, annotations, categories, image_dir, output_dir)

    cleanup(image_dir)
