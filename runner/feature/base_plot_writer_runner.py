from pathlib import Path

from feature.plot_images import PlotWriter
from utils.data import dataset


def run(window_size, stride):
    root = Path("/mnt/gcs/kaggle-vbs-power-line-fault-detection/features")

    root.mkdir(exist_ok=True, parents=True)
    print("writing train data plots....")
    writer = PlotWriter(root.joinpath("train"))
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    writer.write(train_data_set, window_size, stride, figsize=(510, 510))

    print("writing test data plots....")
    writer = PlotWriter(root.joinpath("test"))
    test_data_set = dataset.VbsDataSetFactory()(is_train=False)
    writer.write(test_data_set, window_size, stride, figsize=(510, 510))
    print("done!")
