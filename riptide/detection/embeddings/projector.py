import os
from typing import Any, Callable, Iterable, List

import torch
from sklearn.cluster import DBSCAN
from torchvision.transforms import Compose
from torchvision.transforms.functional import resize
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from riptide.detection.embeddings.mep.encoders import VariableLayerEncoder
from riptide.detection.embeddings.mep.transforms import inverse_normalize, normalize


class CropProjector:
    def __init__(
        self,
        name: str,
        images: torch.Tensor,
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
        label_mask_func: Callable[[list], List[bool]] = None,
        extend: torch.Tensor = None,
    ) -> torch.Tensor:
        """Get embeddings for a given set of labels

        Parameters
        ----------
        labels : Iterable, optional
            Labels to filter by, if None, all embeddings are returned, by default None
        extend : torch.Tensor, optional
            Additional tensor to extend the embeddings by, by default None

        Returns
        -------
        torch.Tensor
            Embeddings for the given labels
        """
        if self._embeddings is None:
            embeddings, _, _ = self._get_embeddings()

            if self.normalize_embeddings:
                embeddings = torch.nan_to_num(
                    (embeddings - embeddings.min(dim=0)[0])
                    / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
                )

            self._embeddings = embeddings

        if label_mask_func is not None:
            mask = label_mask_func(self.labels)
            embeddings = self._embeddings[mask]
        else:
            embeddings = self._embeddings

        if extend is not None:
            assert extend.size(0) == embeddings.size(
                0
            ), "Embedding and extend must have the same number of rows"
            if self.normalize_embeddings:
                extend = torch.nan_to_num(
                    (extend - extend.min(dim=0)[0])
                    / (extend.max(dim=0)[0] - extend.min(dim=0)[0])
                )
            embeddings = torch.cat([embeddings, extend], dim=1)

        if self.normalize_embeddings:
            embeddings = torch.nan_to_num(
                (embeddings - embeddings.min(dim=0)[0])
                / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
            )

        return embeddings

    def cluster(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        label_mask_func: Callable[[list], List[bool]] = None,
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(label_mask_func)
        if len(embeddings) == 0:
            return torch.zeros(0, dtype=torch.long)

        return torch.tensor(
            DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings).labels_
        )
