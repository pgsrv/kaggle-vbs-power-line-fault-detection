from pathlib import Path

from runner.model.plot_model_runner import Vgg16PretrainedTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_800000_stride_800000/")
    save_dir = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/vgg16_pretrained/window_800000_stride_800000_avoid_overfit")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = Vgg16PretrainedTrainRunner(dropout_rate=0.7, train_batch_size=128, valid_batch_size=128,
                                        n_epochs=50, lr=2e-5, weight_decay=1e-4)

    runner(plot_root, save_dir=save_dir)
