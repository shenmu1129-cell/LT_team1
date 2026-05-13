from __future__ import annotations

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(num_classes: int, cfg: dict):
    model_cfg = cfg.get("model", {})
    weights_name = model_cfg.get("weights", "DEFAULT")
    weights = None
    if weights_name and str(weights_name).upper() not in {"NONE", "NULL"}:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=int(model_cfg.get("trainable_backbone_layers", 3)),
        min_size=int(model_cfg.get("min_size", 800)),
        max_size=int(model_cfg.get("max_size", 1333)),
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
