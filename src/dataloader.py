import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

def get_data_loaders(batch_size=4, data_dir="dataset"):
    train_dataset = get_coco_dataset(f"{data_dir}/train", "dataset/train/train.json")
    val_dataset = get_coco_dataset(f"{data_dir}/val", "dataset/val/val.json")
    test_dataset = get_coco_dataset(f"{data_dir}/test", "dataset/test/test.json")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    return train_loader, val_loader, test_loader



