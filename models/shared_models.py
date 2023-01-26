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
from configs.optim_sche_cfg import get_config_optimizers # noqa
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

        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(args.encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        # self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()

        # Config the loss_fn
        # self.loss_fn = loss_fn

    def forward(self, x):
        # x = self.model(x)
        # return x
        # normalize image here #todo
        x = (x - self.mean) / self.std
        mask = self.model(x)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4, f"Error image tensor shape: {image.shape}"

        # Check that image dims are dicisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
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
            self.val_metrics.update(prob_mask, mask.long())

        # tp, fp, tn, fn = smp.metrics.get_stats(
        #     pred_mask.long(), mask.long(), mode="binary"
        # )
        return {
            "loss": loss,
            # "tp": tp,
            # "fp": fp,
            # "fn": fn,
            # "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        # tp = torch.cat([x["tp"] for x in outputs])
        # fp = torch.cat([x["fp"] for x in outputs])
        # fn = torch.cat([x["fn"] for x in outputs])
        # tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        # per_image_iou = smp.metrics.iou_score(
        #     tp, fp, fn, tn, reduction="micro-imagewise"
        # )

        # dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        # dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        # dataset_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        # dataset_bacc = smp.metrics.balanced_accuracy(tp, fp, fn, tn, reduction="micro")
        # dataset_prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        # dataset_rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")

        # metrics = {
        #     f"{stage}/iou": dataset_iou,
        #     f"{stage}/f1": dataset_f1,
        #     f"{stage}/acc": dataset_acc,
        #     f"{stage}/bacc": dataset_bacc,
        #     f"{stage}/rec": dataset_rec,
        #     f"{stage}/prec": dataset_prec,
        # }

        # self.log_dict(metrics, prog_bar=False)

        if stage != "train":
            output = self.val_metrics.compute()
            self.log_dict(output)
            self.val_metrics.reset()
        else:
            self.train_metrics.reset()

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
        return get_config_optimizers(
            optim_name=self.args.optim_name,
            sche_name=self.args.sche_name,
            model=self,  # model is self
        )

    def get_train_val_set(self):
        all_set = get_building_dataset(dataset_path=self.args.dataset_dir)
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
