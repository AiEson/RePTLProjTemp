# Template train.py
import inspect
import os
import sys

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

# Please Change the model LightningModule here ⬇️
from models.UnetPlusPlus import UPPLightningModule  # noqa
from mass_trainers.base.test import test_main  # noqa
from models.encoders import ResNeSt_GSoP_Mean_SCA

if __name__ == "__main__":
    # Then here ⬇️
    l_module = UPPLightningModule
    test_main(l_module=l_module)