import os
from typing import Dict
from urllib.request import urlopen

import torch
from torchvision.models.resnet import BasicBlock, ResNet
from tqdm import tqdm

ALLOWED_MODES = [
    "preconv",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]


def load_state_dict_from_url(name: str) -> Dict[str, torch.Tensor]:
    """Retrieve weights from the Bifrost mise-en-place S3 bucket.

    Args:
        name: 'resnet18_encoder_preconv.pt' (39.6KB) or
            'resnet18_encoder_full.pt' (42.7MB)
    """

    WEIGHTS_NAMES = ["resnet18_encoder_preconv.pt", "resnet18_encoder_full.pt"]
    assert name in WEIGHTS_NAMES, f"'name' must be in {WEIGHTS_NAMES}, got {name}."

    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache")
    state_dict_path = os.path.join(CACHE_DIR, name)
    if os.path.exists(state_dict_path):
        if torch.cuda.is_available():
            return torch.load(state_dict_path)
        else:
            return torch.load(state_dict_path, map_location=torch.device("cpu"))

    BUCKET_ROOT = (
        "https://bifrost-mise-en-place-weights-public.s3.ap-southeast-1.amazonaws.com"
    )
    target = f"{BUCKET_ROOT}/{name}"

    # https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/hub.py#L409 # noqa: E501
    u = urlopen(target)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")

    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    print(f"Downloaded {name} ({file_size} bytes) from {BUCKET_ROOT} to {CACHE_DIR}")

    with tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
        with open(state_dict_path, "w+b") as temp:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                temp.write(buffer)
                pbar.update(len(buffer))
    return torch.load(state_dict_path)


class VariableLayerEncoder(ResNet):
    def __init__(self, mode: str, pretrained=True) -> None:
        super().__init__(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
        )
        del self.fc
        if mode not in ALLOWED_MODES:
            raise Exception(f"mode must be in {ALLOWED_MODES}")

        self.mode = mode
        self.encoder_name = f"resnet18_encoder_{mode}"
        if pretrained:
            self.load_state_dict(load_state_dict_from_url("resnet18_encoder_full.pt"))

        self.preconv = torch.nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mode in ALLOWED_MODES:
            x = getattr(self, mode)(x)
            if self.mode == mode:
                return torch.mean(x, dim=(2, 3))
