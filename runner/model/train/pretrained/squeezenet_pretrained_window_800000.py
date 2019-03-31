from pathlib import Path

from runner.model.plot_model_runner import SqueezenetPretrainedTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_800000_stride_800000/")
    save_dir = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/squeezenet_pretrained/window_800000_stride_800000")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = SqueezenetPretrainedTrainRunner(dropout_rate=0.5, train_batch_size=128, valid_batch_size=128,
                                             n_epochs=70, lr=1e-3, weight_decay=1e-6, patience=20)

    runner(plot_root, save_dir=save_dir)
