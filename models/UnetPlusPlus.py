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
