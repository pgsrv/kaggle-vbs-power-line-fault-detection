from pathlib import Path

from feature.plot_images import FrozenResnet18Feature

if __name__ == '__main__':
    train_plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_800000_stride_800000/")

    test_plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/test/window_800000_stride_800000/")

    print("extracting from train.....")
    train_save_path = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/features/train/resnet18/window_800000_stride_80000")
    train_save_path.mkdir(parents=True, exist_ok=True)

    feature = FrozenResnet18Feature()
    feature.from_plots(train_plot_root, save_path=train_save_path, batch_size=800)

    print("extracting from test.....")
    test_save_path = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/features/test/resnet18/window_800000_stride_80000")
    test_save_path.mkdir(parents=True, exist_ok=True)

    feature.from_plots(test_plot_root, save_path=test_save_path, batch_size=800)
    print("done!!")
