from pathlib import Path

from feature.base import SummaryWindowFeature
from utils.data import dataset


def run(window_size, step_size):
    src_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/aug/")
    save_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/summary/aug/window_{}_step_{}".format(window_size,
                                                                                                  step_size))

    save_root.mkdir(exist_ok=True, parents=True)
    print("writing train summary data ....")
    train_path = save_root.joinpath("train")
    train_path.mkdir(exist_ok=True, parents=True)
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    feature = SummaryWindowFeature(window_size=window_size, step_size=step_size, axis=1)
    feature.transform(src_root, train_path)

    print("done!")