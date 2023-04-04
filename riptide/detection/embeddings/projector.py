import os
from typing import Any, List, Tuple

import PIL.Image as Image
import torch
from mise_en_place.encoders import VariableLayerEncoder
from mise_en_place.objects import COCOObjects
from mise_en_place.projector import _Projector
from mise_en_place.transforms import inverse_normalize, normalize
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
        super().__init__(
            name=name,
            output_dir=output_dir,
            encoder_mode=encoder_mode,
            normalize_embeddings=normalize_embeddings,
            num_categories=1,
            transform=transform,
            device=device,
            write_previews=True,
        )

    def _build_dataset(self) -> Dataset:
        preview_size = self.preview_size

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

    def project(self, write: bool = False) -> None:

        for idx in range(self.num_categories):
            if self.write_previews:
                embeddings, preview, labels = self._get_embeddings(idx=idx)
            else:
                embeddings, labels = self._get_embeddings(idx=idx)
                preview = None

            if self.normalize_embeddings:
                embeddings = torch.nan_to_num(
                    (embeddings - embeddings.min(dim=0)[0])
                    / (embeddings.max(dim=0)[0] - embeddings.min(dim=0)[0])
                )
            if write:
                self.writer.add_embedding(
                    embeddings,
                    metadata=labels,
                    label_img=preview,
                    global_step=idx,
                    tag=str(idx),
                )

        if write:
            self.writer.close()
        else:
            return embeddings, preview, labels
