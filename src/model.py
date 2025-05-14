import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, freeze_backbone=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200,
    max_size=200)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
            
    return model