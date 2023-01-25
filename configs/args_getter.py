import argparse


def get_args():
    parser = argparse.ArgumentParser()

    arg = parser.add_argument

    arg("--dataset_dir", type=str, default="./datasets")
    arg("--precision", type=int, default=32)
    arg("--img_size", type=int, default=512)
    arg("--num_workers", type=int, default=8)
    arg("--project", type=str, default="BuildingSeg")
    arg("--name", type=str, default="DeeplabV3_ResNet34")
    arg("--encoder_name", type=str, default="resnet34")

    arg("--epoch", type=int, default=5)
    arg("--batch_size", type=int, default=16)
    arg("--acc_batch", type=int, default=1)
    arg("--gpus", type=int, default=-1)

    arg('--use_swa', type=bool, default=False)

    return parser.parse_args()
