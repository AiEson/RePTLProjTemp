import albumentations as A


from albumentations.pytorch.transforms import ToTensorV2


class Aaa:
    def __init__(self):
        self.img_size = 123


args = Aaa()


def init_base_transform(args):
    base_transform = []
    if args.img_cropsize != -1:
        base_transform.append(A.RandomCrop(args.img_cropsize, args.img_cropsize))

    return base_transform


base_transform = init_base_transform(args)


print(base_transform)
