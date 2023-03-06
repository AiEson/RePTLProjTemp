import inspect
import sys
from os.path import dirname, join, realpath

import segmentation_models_pytorch as smp

sys.path.append(
    realpath(join(dirname(inspect.getfile(inspect.currentframe())), "../"))
)  # noqa
sys.path.append(
    realpath(join(dirname(inspect.getfile(inspect.currentframe())), "./"))
)  # noqa

from shared_models import PublicSMPModel  # noqa


class UPPLightningModule(PublicSMPModel):
    def __init__(self, args):
        model = smp.UnetPlusPlus(
            encoder_name=args.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        super().__init__(model, args)

# Please Change the model LightningModule here ⬇️
from models.UnetPlusPlus import UPPLightningModule  # noqa
from whu_trainers.base.train import main  # noqa
from models.encoders import ResNeSt_GSoP_Mean_SCA

if __name__ == "__main__":
    # Then here ⬇️
    l_module = UPPLightningModule
    main(l_module=l_module)