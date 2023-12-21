from base import BaseDataLoader
from dataset.star import CustomStarDataset
import albumentations as A


class CustomStarDataLoader(BaseDataLoader):
    def __init__(self, image_dir, mask_dir, is_train,
                 batch_size=1, shuffle=True,
                 validation_split=4, num_workers=1):
        if is_train:
            transforms = A.Compose(
                [
                    A.Resize(height=256, width=256),
                    A.Rotate(limit=35, p=0.4),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                ],
            )
        else:
            transforms = None

        self.dataset = CustomStarDataset(image_dir=image_dir,
                                         mask_dir=mask_dir,
                                         transform=transforms
                                         )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
