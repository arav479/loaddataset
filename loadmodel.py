import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large


MODEL_PATH = r"D:\models\mobilenetv3-large-0.75-9632d2a8.pth"
CLASS_MAPPING_PATH = Path(r"D:\web-pii-detector\pii_classes.json")


def load_pii_classes(path: Path) -> list[str]:
    """Load the class list extracted from the actual WebPII annotations."""
    if not path.is_file():
        raise FileNotFoundError(
            f"PII class mapping not found: {path}. "
            "Run build_pii_classes.py first."
        )
    class_names = json.loads(path.read_text(encoding="utf-8"))["class_names"]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("pii_classes.json must contain a non-empty unique class_names list.")
    return class_names


PII_CLASSES = load_pii_classes(CLASS_MAPPING_PATH)
NUM_OBJECT_CLASSES = len(PII_CLASSES)


class PiiDetector(nn.Module):
    """MobileNetV3 feature backbone followed by a small dense detection head."""

    def __init__(self, checkpoint_path: str, num_object_classes: int = 1):
        super().__init__()

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            name.removeprefix("module."): weights
            for name, weights in state_dict.items()
        }

        # Find the MobileNetV3 width that matches the saved *backbone* weights.
        # The old ImageNet classifier is intentionally excluded: it is replaced
        # below, so its shape cannot prevent the detector from loading.
        best_model = None
        best_width = None
        best_matching_weights = {}
        for width in (0.5, 0.75, 1.0):
            candidate = mobilenet_v3_large(weights=None, width_mult=width)
            candidate_state = candidate.state_dict()
            matching_weights = {
                name: weights
                for name, weights in state_dict.items()
                if name.startswith("features.")
                and name in candidate_state
                and candidate_state[name].shape == weights.shape
            }
            if len(matching_weights) > len(best_matching_weights):
                best_model = candidate
                best_width = width
                best_matching_weights = matching_weights

        if best_model is None or not best_matching_weights:
            raise RuntimeError(
                "The checkpoint contains no MobileNetV3 feature weights matching "
                "Torchvision's mobilenet_v3_large architecture."
            )

        best_model.load_state_dict(best_matching_weights, strict=False)
        print(
            f"Loaded {len(best_matching_weights)} backbone tensors "
            f"using width_mult={best_width}. ImageNet classifier weights were skipped."
        )

        # Keep only convolutional visual features. avgpool and classifier are
        # deliberately not assigned, so the 1,000-class ImageNet output is gone.
        self.backbone = best_model.features

        # Infer the channel count from this particular backbone so it cannot be
        # wrong when a different compatible width is selected above.
        self.backbone.eval()
        with torch.no_grad():
            backbone_channels = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]

        # At 224 x 224 input, this produces a 7 x 7 prediction grid.
        # Per grid cell: objectness, box centre x/y, box width/height, class logits.
        self.detection_head = nn.Conv2d(
            in_channels=backbone_channels,
            out_channels=5 + num_object_classes,
            kernel_size=1,
        )

        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Stage 1: update detection_head only."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Stage 2: fine-tune the pretrained backbone too."""
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return self.detection_head(features)


detector = PiiDetector(MODEL_PATH, num_object_classes=NUM_OBJECT_CLASSES)
print(f"PII classes ({NUM_OBJECT_CLASSES}): {PII_CLASSES}")

# Stage 1 optimizer: only the new detection head has gradients.
stage_1_optimizer = torch.optim.AdamW(
    detector.detection_head.parameters(),
    lr=1e-3,
)

# Example forward pass. Output shape for a 224x224 image: [batch, 6, 7, 7].
example_images = torch.randn(2, 3, 224, 224)
predictions = detector(example_images)
print("Predictions:", predictions.shape)

# After the head begins to learn, begin fine-tuning with a lower learning rate:
# detector.unfreeze_backbone()
# stage_2_optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-4)
