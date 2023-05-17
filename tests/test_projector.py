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

NUM_IMAGES = 10
IMG_SIZE = 32
SEED = 1234


def test_get_embeddings():
    """A CropProjector should be able to be instantiated and compute embeddings"""
    torch.random.manual_seed(SEED)

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


def test_embeddings_are_equal():
    """Embeddings should be equal for the same image"""
    torch.random.manual_seed(SEED)
    images = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(NUM_IMAGES)]

    for mode in ALLOWED_MODES:
        projector = CropProjector(
            name="test",
            images=images + images,
            encoder_mode=mode,
            normalize_embeddings=True,
        )
        embeddings = projector.get_embeddings()
        assert torch.all(
            embeddings[:NUM_IMAGES] == embeddings[NUM_IMAGES:]
        ), "Embeddings should be equal for the same image"


def test_get_clusters():
    """A CropProjector should be able to be instantiated and compute clusters"""
    torch.random.manual_seed(SEED)

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


def test_clusters_are_equal():
    """Clusters should be equal for the same image"""
    torch.random.manual_seed(SEED)
    images = [torch.rand(3, IMG_SIZE, IMG_SIZE) for _ in range(NUM_IMAGES)]

    for mode in ALLOWED_MODES:
        projector = CropProjector(
            name="test",
            images=images + images,
            encoder_mode=mode,
            normalize_embeddings=True,
        )
        clusters = projector.subcluster(sub_lambda=1)
        assert (
            clusters[:, 0].min() != -1
        ), "Core clusters: No outliers should be present"
        # TODO: Fix this test. Identical images should be in the same subcluster
        # assert clusters[:, 1].min() != -1, "Sub-clusters: No outliers should be present"
        assert torch.all(
            clusters[:NUM_IMAGES, 0] == clusters[NUM_IMAGES:, 0]
        ), "Clusters should be equal for the same image"
        assert torch.all(
            clusters[:NUM_IMAGES, 1] == clusters[NUM_IMAGES:, 1]
        ), "Subclusters should be equal for the same image"
