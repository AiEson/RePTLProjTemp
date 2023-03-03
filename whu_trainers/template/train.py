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
from whu_trainers.base.train import main  # noqa

if __name__ == "__main__":
    # Then here ⬇️
    l_module = UPPLightningModule
    main(l_module=l_module)
