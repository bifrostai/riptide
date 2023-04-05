import os
from typing import Any, List, Tuple, Union

import PIL.Image as Image
import torch
from mise_en_place.encoders import VariableLayerEncoder
from mise_en_place.objects import COCOObjects
from mise_en_place.projector import _Projector
from mise_en_place.transforms import inverse_normalize, normalize
from sklearn.cluster import DBSCAN
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, RandomCrop, ToTensor
from torchvision.transforms.functional import crop, pad, resize, to_tensor
from tqdm import tqdm


class CropProjector(_Projector):
    def __init__(
        self,
        name: str,
        images: torch.Tensor,
        output_dir: str,
        encoder_mode: str,
        normalize_embeddings: bool,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.inverse_transform = inverse_normalize()
        transform = Compose([normalize()])
        self.images = images
        self._embeddings: torch.Tensor = None
        super().__init__(
            name=name,
            output_dir=output_dir,
            encoder_mode=encoder_mode,
            normalize_embeddings=normalize_embeddings,
            num_categories=1,
            transform=transform,
            device=device,
            write_previews=False,
        )

    def _build_dataset(self) -> Dataset:
        class CropDataset(Dataset):
            def __init__(
                self,
                images: List[torch.Tensor],
                transform: Compose,
            ) -> None:
                self.images = images
                self.transform = transform

            def __len__(self) -> int:
                return len(self.images)

            def __getitem__(self, idx: int) -> Any:
                return self.images[idx]

            def _collate_fn(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
                thumbnails, inputs = zip(*batch)
                return torch.stack(thumbnails, dim=0), torch.cat(inputs, dim=0)

        return CropDataset(self.images, self.transform)

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

    def get_embeddings(self) -> torch.Tensor:
        if self._embeddings is None:
            embeddings, _, _ = self._get_embeddings()

            if self.normalize_embeddings:
                embeddings = torch.nan_to_num(
                    (embeddings - embeddings.min(dim=0)[0])
                    / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
                )
            self._embeddings = embeddings

        return self._embeddings

    def cluster(self, eps: float = 0.5, min_samples: int = 5) -> torch.Tensor:
        embeddings = self.get_embeddings()
        return torch.tensor(
            DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings).labels_
        )
