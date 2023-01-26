import argparse
import glob
import inspect
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

sys.path.append(
    os.path.realpath(
        os.path.join(
            os.path.dirname(inspect.getfile(inspect.currentframe())), "../../"
        )  # noqa
    )
)
sys.path.append(
    os.path.realpath(
        os.path.join(
            os.path.dirname(inspect.getfile(inspect.currentframe())),
            "../../../",  # noqa
        )
    )
)

from configs.args_getter import get_args  # noqa
from models.DeepLabv3p import DeepLabv3pLightningModule  # noqa


def main() -> None:
    # get train args from get_args() func.
    args = get_args()
    # Set All random seeds
    pl.seed_everything(args.seed)
    # define the model
    model = DeepLabv3pLightningModule(args)

    # ------ Config Train Details -------

    wandb_logger = WandbLogger(project=args.project, name=args.name, group=args.name) # noqa

    # config checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val/BinaryJaccardIndex",
        dirpath="checkpoints",
        filename=f"{args.name}" + "{val/iou:.4f}",
        save_top_k=5,
        mode="max",
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/BinaryJaccardIndex", min_delta=1, patience=50, mode="min"
    )

    callback_list = [checkpoint_callback, early_stop_callback]
    # SWA
    if args.use_swa:
        callback_list.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-4)
        )  # noqa

    # Make Trainer

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        log_every_n_steps=50,
        strategy="ddp_find_unused_parameters_false",
        logger=wandb_logger,
        callbacks=callback_list,
        accumulate_grad_batches=args.acc_batch,
        precision=args.precision,
        max_epochs=args.epochs,
    )
    # Start Train

    trainer.fit(model=model)

    wandb.finish()


if __name__ == "__main__":
    main()
