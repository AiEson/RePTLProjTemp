import inspect
import sys
from os.path import dirname, join, realpath

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(
    realpath(join(dirname(inspect.getfile(inspect.currentframe())), "../"))
)  # noqa
sys.path.append(
    realpath(join(dirname(inspect.getfile(inspect.currentframe())), "./"))
)  # noqa

from configure_optimizers import config_optimizers  # noqa

from configs.loss_cfg import get_loss_result  # noqa
from datasets import get_building_dataset  # noqa


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
                encoder_name=args.encoder_name
                if args is not None
                else "resnet34",  # noqa
                in_channels=3,
                classes=1,
                encoder_weights=None,
            )
        else:
            self.model = model

        # Config the dataset
        self.get_train_val_set()

    def forward(self, x):
        x = self.model(x)
        return x

    def shared_step(self, batch, stage):
        image, mask = batch
        assert image.ndim == 4, f"Error image tensor shape: {image.shape}"

        # Check that image dims are dicisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1,
        # NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self(image)

        # get loss
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, tn, fn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            f"{stage}/loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(
        #     tp, fp, fn, tn, reduction="micro-imagewise"
        # )

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        dataset_bacc = smp.metrics.balanced_accuracy(
            tp, fp, fn, tn, reduction="micro"
        )  # noqa
        dataset_prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        dataset_rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}/iou": dataset_iou,
            f"{stage}/f1": dataset_f1,
            f"{stage}/acc": dataset_acc,
            f"{stage}/bacc": dataset_bacc,
            f"{stage}/rec": dataset_rec,
            f"{stage}/prec": dataset_prec,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return config_optimizers(
            optim_name="adamw", sche_name="coswarmrestart", model=self
        )  # noqa

    def get_train_val_set(self):
        all_set = get_building_dataset(dataset_path=self.args.dataset_path)
        setlen = len(all_set)
        self.train_set, self.val_set = random_split(
            all_set, [int(setlen * 0.8), int(setlen * 0.2)]
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
