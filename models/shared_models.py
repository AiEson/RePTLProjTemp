import inspect
import sys
from os.path import dirname, join, realpath

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(realpath(join(dirname(inspect.getfile(inspect.currentframe())), "../")))
sys.path.append(realpath(join(dirname(inspect.getfile(inspect.currentframe())), "./")))


from configs.loss_cfg import get_loss_result, loss_fn  # noqa
from configs.metrics_cfg import get_metrics_collection  # noqa
from configs.optim_sche_cfg import get_config_optimizers  # noqa
from datasets import get_building_dataset, get_whu_dataset  # noqa


class PublicSMPModel(pl.LightningModule):
    """PublicSMPModel class __init__ method.
    only for segmentation_models_pytorch model.

    Parameters
    ----------
    args : Any
        get configs by args.
    model : smp.base.SegmentationModel
        Custom SMP(segmentation_models_pytorch) model,
        DeepLabV3Plus (ResNet34) as default.
    """

    def __init__(self, model: smp.base.SegmentationModel = None, args=None):

        super().__init__()
        if model is None:
            self.model = smp.DeepLabV3Plus(
                encoder_name=args.encoder_name if args is not None else "resnet34",
                in_channels=3,
                classes=1,
                encoder_weights=None,
            )
        else:
            self.model = model

        self.args = args
        # Config the dataset
        self.get_train_val_set()

        metrics = get_metrics_collection()
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")

        self.use_training_normlization = False

        # preprocessing parameters for image
        if args.encoder_name in smp.encoders.encoders:
            params = smp.encoders.get_preprocessing_params(args.encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
            self.use_training_normlization = True

        print("use_training_normlization: ", self.use_training_normlization)

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # Config the loss_fn
        self.loss_fn = get_loss_result

    def forward(self, x):
        if self.use_training_normlization:
            x = (x - self.mean) / self.std
        mask = self.model(x)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        # print("image.shape: ", image.shape)
        assert image.ndim == 4, f"Error image tensor shape: {image.shape}"

        # Check that image dims are dicisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        # print("mask.shape: ", mask.shape)
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1,
        # NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self(image)

        # get loss
        loss = self.loss_fn(logits_mask, mask.float())

        prob_mask = logits_mask.sigmoid()
        # pred_mask = (prob_mask > 0.5).float()

        if stage == "train":
            output = self.train_metrics(prob_mask, mask.long())
            self.log_dict(output)

        else:
            # self.val_metrics.update(prob_mask, mask.long())
            output = self.val_metrics(prob_mask, mask.long())
            self.log_dict(output)

        return {
            "loss": loss,
        }

    def shared_epoch_end(self, outputs, stage):

        if stage != "train":
            # output = self.val_metrics.compute()
            # self.log_dict(output)
            self.val_metrics.reset()
        else:
            self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        # print("training_step: ", len(batch))
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        # print("validation_step: ", len(batch))
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return get_config_optimizers(
            optim_name=self.args.optim_name,
            sche_name=self.args.sche_name,
            model=self,  # model is self
            base_lr=self.args.lr,
        )

    def get_train_val_set(self):
        if self.args.dataset == "INRIA":
            all_set = get_building_dataset(dataset_path=self.args.dataset_dir)
            setlen = len(all_set)
            self.train_set, self.val_set = random_split(
                all_set, [int(setlen * 0.8), int(setlen * 0.2)]
            )
        elif self.args.dataset == "WHU":
            self.train_set, self.val_set, self.test_set = get_whu_dataset(
                dataset_path=self.args.dataset_dir
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


if __name__ == "__main__":
    a = torch.randn(2, 3, 256, 256)
    model = PublicSMPModel()
    out = model(a)
    print(out.shape)
