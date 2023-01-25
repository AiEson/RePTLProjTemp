import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as D
from torchvision import transforms as T

warnings.filterwarnings("ignore")


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(512, 512)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, transform, test_mode=False, img_size=512):
        self.paths = paths
        self.rles = rles
        self.transform = transform
        self.test_mode = test_mode

        self.len = len(paths)
        self.as_tensor = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
            ]
        )

    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            augments = self.transform(image=img, mask=mask)
            return self.as_tensor(augments["image"]), augments["mask"][None]
        else:
            return self.as_tensor(img), ""

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_building_dataset(dataset_path: str):
    """Get the dataset by dataset_path.

    Parameters
    ----------
    dataset_path : str
        Dataset Path, must contain `train` and `train_mask.csv` int the path.

    Returns
    -------
        Dataset Class (fully, need to spilt to train and val set)

    """
    train_mask = pd.read_csv(
        os.path.join(dataset_path, "train_mask.csv"),
        sep="\t",
        names=["name", "mask"],  # noqa
    )
    train_mask["name"] = train_mask["name"].apply(
        lambda x: str(os.path.join(dataset_path, "train")) + "/" + x
    )

    dataset = TianChiDataset(
        train_mask["name"].values,
        train_mask["mask"].fillna("").values,
        trfm,
        False,
    )

    return dataset
