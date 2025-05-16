import torch
from src.model import get_model
from src.dataloader import get_data_loaders
from src.train import training_loop_with_eval

def run_all_experiments():
    experiments = [
        {"lr": 0.005, "batch_size": 4, "momentum": 0.9, "model_name": "exp5", "freeze_backbone": True},
        {"lr": 0.005, "batch_size": 4, "momentum": 0.9, "model_name": "exp1", "freeze_backbone": False},
        {"lr": 0.001, "batch_size": 4, "momentum": 0.9, "model_name": "exp2", "freeze_backbone": False},
        {"lr": 0.005, "batch_size": 8, "momentum": 0.9, "model_name": "exp3", "freeze_backbone": False},
        {"lr": 0.005, "batch_size": 4, "momentum": 0.8, "model_name": "exp4", "freeze_backbone": False},
        {"lr": 0.005, "batch_size": 4, "momentum": 0.8, "model_name": "exp6", "freeze_backbone": True}
    ]

    num_classes = 5
    train_json = "dataset/train/train.json"
    val_json = "dataset/val/val.json"
    n_epochs = 10

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for idx, exp in enumerate(experiments, start=1):
        print(f"\n[{idx}/{len(experiments)}] Trening eksperymentu: {exp['model_name']}")
        print(f"   - lr={exp['lr']} | batch={exp['batch_size']} | momentum={exp['momentum']} | freeze_backbone={exp['freeze_backbone']}")

        train_loader, val_loader, _ = get_data_loaders(batch_size=exp["batch_size"])
        model = get_model(num_classes=num_classes, freeze_backbone=exp["freeze_backbone"])
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=exp["lr"], momentum=exp["momentum"], weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        training_loop_with_eval(
            model=model,
            data_loader_train=train_loader,
            data_loader_val=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_name=exp["model_name"],
            annotation_file_train=train_json,
            annotation_file_val=val_json,
            n_epochs=n_epochs
        )

    print("\n✅ Wszystkie eksperymenty zakończone.")