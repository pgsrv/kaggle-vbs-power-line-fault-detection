from pathlib import Path

from runner.model.plot_model_runner import XceptionUndersampleBaggingKfoldTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_800000_stride_800000/")
    save_dir = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/xception_pretrained_bagging/window_800000_stride_800000")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = XceptionUndersampleBaggingKfoldTrainRunner(dropout_rate=0.5, train_batch_size=80, valid_batch_size=80,
                                                        n_epochs=50, lr=1e-4, weight_decay=1e-4, patience=10,
                                                        n_fold=10, n_bag=10)

    runner(plot_root, save_dir=save_dir)
