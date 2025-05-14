from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import json
import numpy as np
from tqdm import tqdm
import csv
import torch
import matplotlib.pyplot as plt
import sys
import io

#calculate mean average precision
def evaluate_mAP(model, data_loader, annotation_file):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    coco_gt = COCO(annotation_file)
    results = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluation"):
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                if not targets[i]:
                    continue
                image_id = targets[i][0]["image_id"]
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    results.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(score)
                    })
    with open("temp_predictions.json", "w") as f:
        json.dump(results, f)

    if not results:
        print("No prediction — return metrics of zero.")
        return {k: 0.0 for k in [
            "mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
        ]}

    coco_dt = coco_gt.loadRes("temp_predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    coco_eval.summarize()
    sys.stdout = stdout

    if not hasattr(coco_eval, "stats") or coco_eval.stats is None or len(coco_eval.stats) < 12:
        print("COCOeval not generated full metrics — return 0.0")
        return {k: 0.0 for k in [
            "mAP", "mAP_50", "mAP_75", "mAP_small", "mAP_medium", "mAP_large",
            "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"
        ]}

    metrics = {
        "mAP": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11]
    }

    return metrics

def save_map_to_csv(model_name, map_train, map_val):
    with open(f"results/{model_name}.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "Train mAP", "Validation mAP"])
        for i, (m_train, m_val) in enumerate(zip(map_train, map_val), start=1):
            writer.writerow([f"epoch_{i}", m_train, m_val])
    print(f"Results_saved: {model_name}")

def save_map_plot(model_name, map_train, map_val):
    # plot mAP
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(map_train) + 1), map_train, marker='o', label='Train mAP')
    plt.plot(range(1, len(map_val) + 1), map_val, marker='s', label='Validation mAP')
    plt.title(f"{model_name}: mAP (IoU=0.5:0.95) in different epochs")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{model_name}_plot.png")