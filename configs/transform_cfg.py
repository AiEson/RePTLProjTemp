import albumentations as A


def get_train_transform(img_size=512):
    trfm = A.Compose(
        [
            A.Resize(img_size, img_size),
            # A.RandomCrop(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(),
            A.OneOf(
                [
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                    A.ColorJitter(
                        brightness=0.07,
                        contrast=0.07,
                        saturation=0.1,
                        hue=0.1,
                        always_apply=False,
                        p=0.3,
                    ),
                ],
                p=0.3,
            ),
        ]
    )
    return trfm


def get_base_transform():
    pass
