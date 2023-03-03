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

from MSLNet_git.nets.MSLNet import MSLNet # noqa
from shared_models import PublicSMPModel  # noqa


class MSLNetLightningModule(PublicSMPModel):
    def __init__(self, args):
        model = MSLNet(num_classes=1, pretrained=False)
        super().__init__(model, args)
