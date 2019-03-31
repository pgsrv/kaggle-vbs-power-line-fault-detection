from pathlib import Path

from feature.base import AverageWindowFeature
from utils.data import dataset


def run(window_size, step_size):
    root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/avg/window_{}_step_{}".format(window_size,
                                                                                          step_size))

    root.mkdir(exist_ok=True, parents=True)
    print("writing train avg data ....")
    train_path = root.joinpath("train")
    train_path.mkdir(exist_ok=True, parents=True)
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    feature = AverageWindowFeature(window_size=window_size, step_size=step_size)
    feature.transform(train_data_set, train_path)

    print("writing test avg data....")
    test_data_set = dataset.VbsDataSetFactory()(is_train=False)
    test_path = root.joinpath("test")
    test_path.mkdir(exist_ok=True, parents=True)
    feature.transform(test_data_set, test_path)

    print("done!")
