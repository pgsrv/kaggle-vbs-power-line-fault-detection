from pathlib import Path

from feature.base import PercentileSummaryWindowFeature
from utils.data import dataset


def run():
    root = Path("/mnt/share/vbs-power-line-fault-detection/features/percentile_summary/window_40_step_20")

    root.mkdir(exist_ok=True, parents=True)
    print("writing train summary data ....")
    train_path = root.joinpath("train")
    train_path.mkdir(exist_ok=True, parents=True)
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    feature = PercentileSummaryWindowFeature(window_size=40, step_size=20)
    feature.transform(train_data_set, train_path)

    print("writing test summary data....")
    test_data_set = dataset.VbsDataSetFactory()(is_train=False)
    test_path = root.joinpath("test")
    test_path.mkdir(exist_ok=True, parents=True)
    feature.transform(test_data_set, test_path)

    print("done!")


if __name__ == '__main__':
    run()
