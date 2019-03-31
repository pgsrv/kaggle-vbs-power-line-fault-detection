from pathlib import Path

from runner.model.plot_model_runner import ResNet18KfoldTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_10000_stride_5000/")
    save_dir = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/resnet18_pretrained_kfold/window_10000_stride_5000")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = ResNet18KfoldTrainRunner(dropout_rate=0.5, train_batch_size=128, valid_batch_size=128,
                                      n_epochs=50, lr=1e-4, weight_decay=1e-4, patience=10,
                                      n_fold=10)

    runner(plot_root, save_dir=save_dir)
