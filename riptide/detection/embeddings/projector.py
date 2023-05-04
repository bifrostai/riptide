from __future__ import annotations

from typing import Any, List

import torch
from hdbscan import HDBSCAN
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

        self._clusterer = None
        self._mask = None

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
        hw_ratios = torch.nan_to_num(image_sizes[:, 0] / image_sizes[:, 1]).pow(2)
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

    def get_clusterer(
        self,
        *,
        eps: float = 0.3,
        min_cluster_size: int = 2,
        mask: List[bool] = None,
        **kwargs,
    ) -> HDBSCAN:
        embeddings = self.get_embeddings()
        if mask is not None:
            embeddings = embeddings[mask]

        if (
            self._clusterer is None
            or self._clusterer.min_cluster_size != min_cluster_size
            or self._clusterer.cluster_selection_epsilon != eps
        ):
            kwargs["prediction_data"] = True
            self._clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=eps,
                **kwargs,
            ).fit(embeddings)
        elif self._mask != mask:
            self._clusterer.fit(embeddings)

        self._mask = mask

        return self._clusterer

    def cluster(self, **kwargs) -> torch.Tensor:
        """Cluster embeddings. Provide a mask to perform clustering on a subset of embeddings.

        Parameters
        ----------
        eps : float, optional
            DBSCAN eps parameter, by default 0.4
        min_samples : int, optional
            DBSCAN min_samples parameter, by default 2
        mask : List[bool], optional
            Mask to apply to embeddings, by default None
        **kwargs
            Additional arguments to pass to HDBSCAN

        Returns
        -------
        torch.Tensor
            Cluster labels
        """

        return torch.tensor(self.get_clusterer(**kwargs).labels_)

    def subcluster(self, **kwargs) -> torch.Tensor:
        """Subdivide clusters into subclusters"""
        clusterer = self.get_clusterer(**kwargs)
        embeddings = self.get_embeddings()
        if self._mask is not None:
            embeddings = embeddings[self._mask]

        labels = torch.tensor(clusterer.labels_)
        subclusters = torch.full((embeddings.shape[0],), -1, dtype=torch.long)

        for cluster in range(clusterer.labels_.max() + 1):
            cluster_mask = clusterer.labels_ == cluster
            cluster_embeddings = embeddings[cluster_mask]
            subclusterer = HDBSCAN(min_cluster_size=2, min_samples=1).fit(
                cluster_embeddings
            )
            subclusters[cluster_mask] = torch.tensor(subclusterer.labels_)

        return torch.stack([labels, subclusters], dim=1)

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
