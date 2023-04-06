import os
from typing import Any, Iterable, List, Tuple, Union

import torch
from mise_en_place.encoders import VariableLayerEncoder
from mise_en_place.projector import _Projector
from mise_en_place.transforms import inverse_normalize, normalize
from sklearn.cluster import DBSCAN
from torchvision.transforms import Compose
from torchvision.transforms.functional import crop, pad, resize, to_tensor
from tqdm import tqdm


class CropProjector(_Projector):
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

        for image in tqdm(self.images):
            image = self.transform(image.float())
            instance = image.unsqueeze(0)

            with torch.no_grad():
                embeddings.append(self.encoder(instance.to(self.device)).squeeze(0))

            instance = resize(instance, (self.preview_size, self.preview_size))
            preview.append(self.inverse_transform(instance.squeeze(0)))

        embeddings = torch.stack(embeddings, dim=0)
        preview = torch.stack(preview, dim=0)

        return embeddings, preview, None

    def get_embeddings(self, labels: Iterable = None) -> torch.Tensor:
        if self._embeddings is None:
            embeddings, _, _ = self._get_embeddings()

            if self.normalize_embeddings:
                embeddings = torch.nan_to_num(
                    (embeddings - embeddings.min(dim=0)[0])
                    / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
                )
            self._embeddings = embeddings

        if labels is not None:
            if not isinstance(labels, Iterable):
                labels = [labels]
            mask = [label in labels for label in self.labels]
            return self._embeddings[mask]

        return self._embeddings

    def cluster(
        self, eps: float = 0.5, min_samples: int = 5, by_labels: List = None
    ) -> torch.Tensor:
        embeddings = self.get_embeddings(by_labels)
        if len(embeddings) == 0:
            return torch.zeros(0, dtype=torch.long)

        return torch.tensor(
            DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings).labels_
        )
