# import os

import torch
from torchvision.ops import box_iou

from riptide.detection.embeddings.projector import CropProjector

ALLOWED_MODES = [
    "preconv",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
]


def test_get_embeddings():
    """A CropProjector should be able to be instantiated and compute embeddings"""
    torch.random.manual_seed(0)
    NUM_IMAGES = 10
    IMG_SIZE = 32

    images = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(NUM_IMAGES)]
    for mode in ALLOWED_MODES:
        try:
            projector = CropProjector(
                name="test",
                images=images,
                encoder_mode=mode,
                normalize_embeddings=True,
            )
            projector.get_embeddings()
        except Exception as e:
            raise AssertionError(
                f"Computing embeddings with {NUM_IMAGES} images failed with mode {mode}"
            ) from e

        try:
            projector = CropProjector(
                name="test",
                images=[images[0]],
                encoder_mode=mode,
                normalize_embeddings=True,
            )
            projector.get_embeddings()
        except Exception as e:
            raise AssertionError(
                f"Computing embeddings with a single image failed with mode {mode}"
            ) from e


def test_get_clusters():
    """A CropProjector should be able to be instantiated and compute clusters"""
    torch.random.manual_seed(0)
    NUM_IMAGES = 10
    IMG_SIZE = 32

    images = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(NUM_IMAGES)]
    for mode in ALLOWED_MODES:
        projector = CropProjector(
            name="test",
            images=images,
            encoder_mode=mode,
            normalize_embeddings=True,
        )
        clusters = projector.cluster()
        assert clusters.shape == (NUM_IMAGES,)

        subclusters = projector.subcluster()
        assert subclusters.shape == (NUM_IMAGES, 2)

        assert torch.all(
            projector.subcluster()[:, 0] == projector.cluster()
        ), "First subcluster should be equal to the cluster"
