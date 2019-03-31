from pathlib import Path

from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionPathTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/percentile_summary/aug/window_{}_step_{}/train".format(5000,
                                                                                                                   5000))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/aug/percentile_summary_window_{}_step_{}".format(
            5000, 5000))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VbsDataSetFactory()(is_train=True)

    params = GruParams(
        seq_len=VbsDataSetFactory.SIGNAL_LENGTH // 5000,
        input_size=57,
        hidden_size=96,
        gru_dropout=0.3,
        dense_output=64,
        dropout=0.3
    )

    runner = BiGruAttensionPathTrainRunner(params,
                                           train_batch_size=240, valid_batch_size=240,
                                           n_epochs=200, lr=2e-3, weight_decay=1e-5, patience=40,
                                           n_fold=5, aug_ratio=3)

    runner(train_dataset, saved_feature_root, save_dir=save_dir, validation_metric="loss")
