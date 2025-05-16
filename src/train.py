import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.test import evaluate_mAP, save_map_to_csv, save_map_plot

def train_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in tqdm(data_loader, desc=f"Epoch trained {epoch}"):
        images = [img.to(device) for img in images]

        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                x, y, w, h = obj["bbox"]
                if w > 0 and h > 0:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(obj["category_id"])
            if boxes:
                processed_targets.append({
                    "boxes": torch.tensor(boxes).float().to(device),
                    "labels": torch.tensor(labels).long().to(device)
                })
                valid_images.append(images[i])

        if not processed_targets:
            continue

        loss_dict = model(valid_images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")

def training_loop(model, train_loader, optimizer, scheduler, device, model_name, n_epochs = 10):
    for epoch in range(n_epochs):
        train_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()

        path = f"models/{model_name}_fasterrcnn_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), path)
        print(f"Model saved: {path}")


def training_loop_with_eval(model, data_loader_train, data_loader_val, optimizer, scheduler, device, model_name, annotation_file_train, annotation_file_val, n_epochs = 10):
    map_train = []
    map_val = []

    for epoch in range(n_epochs):
        train_epoch(model, optimizer, data_loader_train, device, epoch)
        scheduler.step()

        path = f"models/{model_name}_fasterrcnn_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), path)
        print(f"Model saved: {path}")

        # Evaluation on train
        metrics_train = evaluate_mAP(model, data_loader_train, annotation_file_train)
        current_map_train = metrics_train["mAP"]
        map_train.append(current_map_train)
        print(f"[Train] mAP: {current_map_train:.4f}")

        # Evaluation on val
        metrics_val = evaluate_mAP(model, data_loader_val, annotation_file_val)
        current_map_val = metrics_val["mAP"]
        map_val.append(current_map_val)
        print(f"[Val]   mAP: {current_map_val:.4f}")

    save_map_to_csv(model_name, map_train, map_val)

    save_map_plot(model_name, map_train, map_val)
