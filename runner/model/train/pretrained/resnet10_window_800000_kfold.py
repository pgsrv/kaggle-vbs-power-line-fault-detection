from pathlib import Path

from model.nn_model_wrapper import DefaultTransformers
from runner.model.plot_model_runner import ResNet10KfoldTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "/mnt/share/vbs-power-line-fault-detection/features/train/window_800000_stride_800000/")
    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/resnet10_kfold/window_800000_stride_800000")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = ResNet10KfoldTrainRunner(dropout_rate=0.5, train_batch_size=512, valid_batch_size=512,
                                      n_epochs=70, lr=1e-4, weight_decay=1e-4, patience=10,
                                      n_fold=10)

    runner(plot_root, save_dir=save_dir, transformers=DefaultTransformers())
