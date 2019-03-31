from pathlib import Path

from feature.base import PhaseStandardScaled
from utils.data import dataset


def run():
    root = Path("/mnt/share/vbs-power-line-fault-detection/features/standardscaled/")

    root.mkdir(exist_ok=True, parents=True)
    print("writing train normalized data ....")
    train_path = root.joinpath("train")
    train_path.mkdir(exist_ok=True, parents=True)
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    normalizer = PhaseStandardScaled()
    normalizer.fit_transform(train_data_set, train_path)

    print("writing test normalized data....")
    test_data_set = dataset.VbsDataSetFactory()(is_train=False)
    test_path = root.joinpath("test")
    test_path.mkdir(exist_ok=True, parents=True)
    normalizer.transform(test_data_set, test_path)

    print("done!")


if __name__ == '__main__':
    run()
