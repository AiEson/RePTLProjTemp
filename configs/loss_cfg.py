import torch.nn as nn
import torch
from segmentation_models_pytorch import losses as smploss

"""
def the loss dict with format: `loss_fn: [factor: float, need_sigmoid: bool]`
"""
_LOSS_DICT = {
    smploss.SoftBCEWithLogitsLoss(smooth_factor=0.1): [0.8, False],
    smploss.JaccardLoss(mode=smploss.BINARY_MODE, smooth=0.1): [0.2, False],
}


def get_loss_result(y_pred, y_true, use_long_as_label=False):
    """
    get_loss_result: get the loss value

    y_pred: predict mask
    y_true: real mask
    use_long_as_label: convert `y_true` to long
    """
    if use_long_as_label:
        y_true = y_true.long()

    ret = None

    for loss_fns, cnfs in _LOSS_DICT.items():
        loss = loss_fns(y_pred if not cnfs[1] else y_pred.sigmoid(), y_true)
        if ret is not None:
            ret = ret + loss * cnfs[0]
        else:
            ret = loss * cnfs[0]

    return ret


if __name__ == "__main__":
    a = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 1, 64, 64)
    print(get_loss_result(a, b))
