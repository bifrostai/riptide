from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List

import torch
from sklearn.cluster import DBSCAN
from torchvision.transforms import Compose
from torchvision.transforms.functional import resize
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from riptide.detection.embeddings.mep.encoders import VariableLayerEncoder
from riptide.detection.embeddings.mep.transforms import inverse_normalize, normalize
from riptide.utils.logging import logger


class CropProjector:
    @logger("Initializing CropProjector", "Initialized CropProjector")
    def __init__(
        self,
        name: str,
        images: List[torch.Tensor],
        encoder_mode: str,
        normalize_embeddings: bool,
        labels: List[Any] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if labels is not None:
            assert len(images) == len(labels), "Number of images and labels must match"

        self.inverse_transform = inverse_normalize()
        transform = Compose([normalize()])
        self.images = images
        self.labels = labels
        self._embeddings: torch.Tensor = None

        # Override __init__: No SummaryWriter
        self.name = name
        self.device = device
        self.num_categories = 1
        self.transform = transform
        self.write_previews = False
        self.normalize_embeddings = normalize_embeddings
        self.encoder = VariableLayerEncoder(mode=encoder_mode).to(device)
        self.preview_size = 32
        self.embedding_size = self.encoder(torch.empty((1, 3, 48, 48)).to(device)).size(
            1
        )

    def project(self):
        raise NotImplementedError(
            "Exporting to Tensorboard is not currently supported for CropProjector"
        )

    def _get_embeddings(self, *args, **kwargs) -> tuple:
        embeddings = list()
        preview = list()

        with logging_redirect_tqdm():
            for image in tqdm(self.images, desc=f"Computing embeddings ({self.name})"):
                image = self.transform(image.float())
                instance = image.unsqueeze(0)

                with torch.no_grad():
                    embeddings.append(self.encoder(instance.to(self.device)).squeeze(0))

                instance = resize(instance, (self.preview_size, self.preview_size))
                preview.append(self.inverse_transform(instance.squeeze(0)))

        embeddings = torch.stack(embeddings, dim=0)
        preview = torch.stack(preview, dim=0)

        image_sizes = torch.tensor([image.shape[-2:] for image in self.images])
        hw_ratios = torch.nan_to_num(image_sizes[:, 0] / image_sizes[:, 1])
        embeddings = torch.cat([embeddings, hw_ratios.unsqueeze(1)], dim=1)

        return embeddings, preview, None

    def get_embeddings(
        self,
    ) -> torch.Tensor:
        """Get embeddings for images

        Returns
        -------
        torch.Tensor
            Embeddings for images
        """
        if self._embeddings is None:
            embeddings, _, _ = self._get_embeddings()

            if self.normalize_embeddings:
                embeddings = torch.nan_to_num(
                    (embeddings - embeddings.min(dim=0)[0])
                    / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
                )

            self._embeddings = embeddings

        return self._embeddings

    def cluster(
        self,
        eps: float = 0.4,
        min_samples: int = 2,
        mask: List[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Cluster embeddings. Provide a mask to perform clustering on a subset of embeddings.

        Parameters
        ----------
        eps : float, optional
            DBSCAN eps parameter, by default 0.4
        min_samples : int, optional
            DBSCAN min_samples parameter, by default 2
        mask : List[bool], optional
            Mask to apply to embeddings, by default None

        Returns
        -------
        torch.Tensor
            Cluster labels
        """
        embeddings = self.get_embeddings()
        if mask is not None:
            embeddings = embeddings[mask]
        if len(embeddings) == 0:
            return torch.zeros(0, dtype=torch.long)

        return torch.tensor(
            DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings).labels_
        )

    def match_clusters(
        self, labels: list, eps: float = 0.4, min_samples: int = 2
    ) -> List[torch.Tensor]:
        """Compute clusters for a subset of labels

        Parameters
        ----------
        labels : list
            List of labels to match
        eps : float, optional
            DBSCAN eps parameter, by default 0.4
        min_samples : int, optional
            DBSCAN min_samples parameter, by default 2

        Returns
        -------
        List[torch.Tensor]
            List of cluster labels for each label
        """
        labels = list(set(labels))

        mask = []
        ids = []
        for label in self.labels:
            mask.append(label in labels)
            if label in labels:
                ids.append(labels.index(label))

        clusters = self.cluster(eps=eps, min_samples=min_samples, mask=mask)
        cluster_groups = [[] for _ in labels]
        for i, cluster in enumerate(clusters):
            if ids[i] == -1:
                continue
            cluster_groups[ids[i]].append(cluster)

        cluster_list = [torch.tensor(cluster_group) for cluster_group in cluster_groups]

        return cluster_list
