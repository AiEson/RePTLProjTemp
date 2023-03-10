from cgi import test
import os
from PIL import Image
import warnings

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as D
from torchvision import transforms as T

from configs.transform_cfg import get_train_transform

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
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]  # noqa
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
            return {
                "image": self.as_tensor(augments["image"]),
                "mask": augments["mask"][None],
            }
        else:
            return {"image": self.as_tensor(augments["image"]), "mask": ""}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


def get_building_dataset(dataset_path: str, img_size=512):
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
        get_train_transform(img_size),
        False,
    )

    return dataset

class WHUDataset(D.Dataset):
    def __init__(self, paths, transform=None, test_mode=False, img_size=512, label_folder_name='label', image_folder_name='image'):
        self.paths = paths
        self.transform = transform
        self.test_mode = test_mode
        self.label_folder_name = label_folder_name
        self.image_folder_name = image_folder_name

        self.len = len(paths)
        self.as_tensor = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                # T.Normalize([0.4352682576428411, 0.44523221318154493, 0.41307610541534784], [0.026973196780331585, 0.026424642808887323, 0.02791246590291434]),
                T.Normalize([0.625, 0.448, 0.688], [0.131, 0.177, 0.101]),
                # T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    # get data operation
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        # mask ??????????????????
        mask = Image.open(self.paths[index].replace(self.image_folder_name, self.label_folder_name))
        # convert PIL.Image
        img = img.convert('RGB')
        mask = mask.convert('L')
        # to numpy
        img = np.array(img)
        mask = np.array(mask)
        # ????????????
        # mask = mask[None]
        # write
        # cv2.imwrite('test.png', img) 
        # cv2.imwrite('test_mask.png', mask * 255)
        if not self.test_mode:
            augments = self.transform(image=img, mask=mask) if self.transform else {'image': img, 'mask': mask}
            # cv2.imwrite('test_aug.png', augments['image'])
            # cv2.imwrite('test_aug_mask.png', augments['mask'] * 255)
            # ??????augments["mask"]?????????
            return {
                "image": self.as_tensor(augments["image"]),
                "mask": T.ToTensor()(augments["mask"]),
            }
        else:
            # print(mask[None].shape)
            return {"image": self.as_tensor(img), "mask": T.ToTensor()(mask)}

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def get_whu_dataset(dataset_path: str, img_size=512):
    """Get the dataset by dataset_path.

    Parameters
    ----------
    dataset_path : str
        Dataset Path, must contain `train` and `train_mask.csv` int the path.

    Returns
    -------
        (train_set, val_set, test_set)

    """
    train_path = os.path.join(dataset_path, "train", 'image')
    val_path = os.path.join(dataset_path, "val", 'image')
    test_path = os.path.join(dataset_path, "test", 'image')
    
    train_set = WHUDataset(
        paths = [os.path.join(train_path, x) for x in os.listdir(train_path)],
        transform=get_train_transform(img_size),
        test_mode=False,
        img_size=img_size
    )
    
    val_set = WHUDataset(
        paths = [os.path.join(val_path, x) for x in os.listdir(val_path)],
        test_mode=False,
        img_size=img_size
    )
    
    test_set = WHUDataset(
        paths = [os.path.join(test_path, x) for x in os.listdir(test_path)],
        test_mode=True,
        img_size=img_size
    )
    

    return train_set, val_set, test_set


def get_massachusetts_dataset(dataset_path: str, img_size=512):
    """Get the dataset by dataset_path.

    Parameters
    ----------
    dataset_path : str
        Dataset Path, must contain `train` and `train_mask.csv` int the path.

    Returns
    -------
        (train_set, val_set, test_set)

    """
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")
    
    train_set = WHUDataset(
        paths = [os.path.join(train_path, x) for x in os.listdir(train_path)],
        transform=get_train_transform(img_size),
        test_mode=False,
        img_size=img_size,
        image_folder_name='train',
        label_folder_name='train_labels'
    )
    
    val_set = WHUDataset(
        paths = [os.path.join(val_path, x) for x in os.listdir(val_path)],
        test_mode=False,
        img_size=img_size,
        image_folder_name='val',
        label_folder_name='val_labels'
    )
    
    test_set = WHUDataset(
        paths = [os.path.join(test_path, x) for x in os.listdir(test_path)],
        test_mode=True,
        img_size=img_size,
        image_folder_name='test',
        label_folder_name='test_labels'
    )
    

    return train_set, val_set, test_set

if __name__ == "__main__":
    dataset_root = '/home/zhaobinguet/codes/datasets/WHU'
    train_set, val_set, test_set = get_whu_dataset(dataset_root)
    
    print(len(train_set), len(val_set), len(test_set))
    print(train_set[0]['image'].shape, train_set[0]['mask'].shape)
    print(train_set[0]['image'].max(), train_set[0]['image'].min(), train_set[0]['mask'].max(), train_set[0]['mask'].min())
    print(val_set[0]['image'].max(), val_set[0]['image'].min(), val_set[100]['mask'].max(), val_set[0]['mask'].min())
    # ----------------------------------------------
    # dataset_root = '/home/zhaobinguet/codes/datasets/buildingSegDataset' 
    # dataset = get_building_dataset(dataset_root)
    # print(len(dataset))
    # print(dataset[0]['image'].shape, dataset[0]['mask'].shape)
    # print(dataset[0]['image'].max(), dataset[0]['image'].min(), dataset[0]['mask'].max(), dataset[0]['mask'].min())      
    # ----------------------------------------------
    # dataset_root = '/home/aieson/codes/datasets/Massachusetts_cropped512'
    # train_set, val_set, test_set = get_massachusetts_dataset(dataset_root)
    
    # print(len(train_set), len(val_set), len(test_set))
    # print(train_set[10]['image'].shape, train_set[10]['mask'].shape)
    # print(train_set[10]['image'].max(), train_set[10]['image'].min(), train_set[10]['mask'].max(), train_set[10]['mask'].min())

    
    