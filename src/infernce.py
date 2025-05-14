import torch
import matplotlib.pyplot as plt
from PIL import Image
from src.model import get_model
import torchvision.transforms.functional as F
import os

# === Config ===
COCO_CLASSES = {
    0: "Background",
    1: "Coyote",
    2: "Deer",
    3: "Hog"
}

def load_model(model_path, num_classes, device, freeze_backbone=False):
    model = get_model(num_classes, freeze_backbone)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def prepare_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor.to(device), image

def get_class_name(class_id):
    return COCO_CLASSES.get(class_id, "Unknown")

def draw_boxes(image, prediction, threshold=0.5, fig_size=(10, 10)):
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    plt.figure(figsize=fig_size)
    plt.imshow(image)
    ax = plt.gca()
    detections_drawn = 0

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = box
            class_name = get_class_name(label)
            ax.add_patch(plt.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=2, edgecolor='r', facecolor='none'
            ))
            ax.text(x_min, y_min, f"{class_name} ({score:.2f})", color='r', fontsize=10)
            detections_drawn += 1

    if detections_drawn == 0:
        print("Nie wykryto żadnych obiektów")

    plt.axis('off')
    plt.show()
    plt.close()

def run_pipeline(image_path, model_path, num_classes=4, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, num_classes, device)

    image_tensor, image_pil = prepare_image(image_path, device)

    with torch.no_grad():
        prediction = model(image_tensor)

    draw_boxes(image_pil, prediction, threshold=threshold)