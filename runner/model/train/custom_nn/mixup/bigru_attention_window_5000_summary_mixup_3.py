from pathlib import Path

from feature.base import PercentileSummaryWindowFeature
from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionMixupTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/percentile_summary/"
                              "window_{}_step_{}".format(5000, 5000))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/mixup/window_5000_summary_mixup_3")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = PercentileSummaryWindowFeature(5000, 5000)

    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array
    train_dataset = VbsDataSetFactory()()

    params = GruParams(
        seq_len=VbsDataSetFactory.SIGNAL_LENGTH // 5000,
        input_size=X.shape[-1],
        hidden_size=128,
        gru_dropout=0.3,
        dense_output=64,
        dropout=0.3
    )
    runner = BiGruAttensionMixupTrainRunner(params,
                                            train_batch_size=256, valid_batch_size=256,
                                            n_epochs=300, lr=2e-3, weight_decay=1e-5, patience=70,
                                            n_fold=5, mixup_ratio=3)

    runner(train_dataset, X, save_dir=save_dir, validation_metric="loss")
