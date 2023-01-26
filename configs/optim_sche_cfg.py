import torch
import torch.optim as topt
import torch.optim.lr_scheduler as tsch

G_LR = 1e-4
G_EPS = 1e-6

"""
Config the All Optimizers by Dict
"""
G_WD = 1e-3

_OPTS_DICT = {
    "adam": [topt.Adam, {"lr": G_LR, "eps": G_EPS}],
    "adamw": [topt.AdamW, {"lr": G_LR, "eps": G_EPS}],
    "adadelta": [topt.Adadelta, {"lr": G_LR, "weight_decay": G_WD}],
    "sgd": [topt.SGD, {"lr": G_LR, "weight_decay": G_WD}],
    "nadam": [
        topt.NAdam,
        {
            "lr": G_LR,
            "eps": G_EPS,
            "nesterov": True,
            "momentum": 0.9,
            "dampening": 0,
            "weight_decay": G_WD,
        },
    ],
    "asgd": [topt.ASGD, {"lr": G_LR, "eps": G_EPS}],
}

"""
Config the All lr_scheduler By Dict
"""
_SCHE_DICT = {
    "reducelr": [
        tsch.ReduceLROnPlateau,
        {
            "patience": 5,
            "factor": 0.5,
            "mode": "max",
        },
    ],
    "cos": [tsch.CosineAnnealingLR, {"T_max": 100}],
    "coswarmrestart": [
        tsch.CosineAnnealingWarmRestarts,
        {"T_0": 5, "T_mult": 1, "eta_min": G_EPS, "last_epoch": -1},
    ],
}


def get_config_optimizers(optim_name: str, model, sche_name: str = None):
    optimizer = _OPTS_DICT[optim_name][0](
        model.parameters(), **_OPTS_DICT[optim_name][1]
    )
    scheduler = None
    if sche_name is None:
        print("No lr_scheduler selected!")
    else:
        scheduler = _SCHE_DICT[sche_name][0](
            optimizer=optimizer, **_SCHE_DICT[sche_name][1]
        )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3(encoder_weights=None, classes=1)
    a = config_optimizers("adamw", model=model, sche_name="coswarmrestart")
    print(a)
