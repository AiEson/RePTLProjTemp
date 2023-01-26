import torch.nn as nn
import torch
from segmentation_models_pytorch import losses as smploss
import segmentation_models_pytorch as smp

"""
def the loss dict with format: `loss_fn: [factor: float, need_sigmoid: bool]`
"""
_LOSS_DICT = {
    smploss.SoftBCEWithLogitsLoss(): [1, False],
    # smploss.JaccardLoss(mode=smploss.BINARY_MODE, smooth=0.1): [0.2, True],
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


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


bce_fn = nn.BCEWithLogitsLoss()
dice_fn = SoftDiceLoss()

softbce_fn = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(y_pred, y_true):
    softbce = softbce_fn(y_pred, y_true)
    # dice = dice_fn(y_pred.sigmoid(), y_true)
    # return 0.8*bce + 0.2*dice
    return softbce


if __name__ == "__main__":
    a = torch.randn(1, 1, 64, 64)
    b = torch.randn(1, 1, 64, 64)
    print(get_loss_result(a, b))
