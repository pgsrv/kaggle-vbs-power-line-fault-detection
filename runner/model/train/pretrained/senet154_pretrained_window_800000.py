from pathlib import Path

from runner.model.plot_model_runner import SeNet154PretrainedTrainRunner

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/train/window_800000_stride_800000/")
    save_dir = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/senet154_pretrained/window_800000_stride_800000")
    save_dir.mkdir(parents=True, exist_ok=True)

    runner = SeNet154PretrainedTrainRunner(dropout_rate=0.5, train_batch_size=16, valid_batch_size=16,
                                           n_epochs=70, lr=1e-4, weight_decay=1e-4, patience=20)

    runner(plot_root, save_dir=save_dir)
