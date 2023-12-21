import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage import io

class CustomStarDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.tif', '_mask.tif'))

        # image = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        # mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        image = np.array(io.imread(img_path).astype('float32'))
        mask = np.array(io.imread(mask_path).astype('float32'))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = torch.from_numpy(image).unsqueeze(dim=0)
        mask = torch.from_numpy(mask).unsqueeze(dim=0)

        return image, mask

    def __len__(self):
        return len(self.images)


def test():
    import albumentations as A

    train_transform = A.Compose(
        [
            A.Resize(height=256, width=256),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
        ],
    )

    train_dir = 'E:/data/telescope/1-7/dataset/train/image/'
    mask_dir = 'E:/data/telescope/1-7/dataset/train/mask/'

    train_ds = CustomStarDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    image, mask = next(iter(train_ds))

    img = torch.squeeze(image).cpu().numpy()
    mask = torch.squeeze(mask).cpu().numpy()

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    test()
