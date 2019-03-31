from pathlib import Path

from feature.base import PercentileSummaryWindowFeature
from model.custom_nn import GruParams
from runner.model.base_runner import BiLstmAttensionTrainRunner
from utils.data.dataset import VbsDataSetFactory

#
# cudnn.enabled = False

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/percentile_summary/"
                              "window_{}_step_{}".format(5000, 5000))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bilstm_attention/window_5000_summary_adabound_original_param")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = PercentileSummaryWindowFeature(5000, 5000)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array
    train_dataset = VbsDataSetFactory()()

    params = GruParams(
        seq_len=VbsDataSetFactory.SIGNAL_LENGTH // 5000,
        input_size=X.shape[-1],
        hidden_size=128,
        gru_dropout=0,
        dense_output=64,
        dropout=0.1
    )
    runner = BiLstmAttensionTrainRunner(params,
                                        train_batch_size=256, valid_batch_size=256,
                                        n_epochs=300, lr=2e-3, weight_decay=0, patience=200,
                                        n_fold=5)

    runner(train_dataset, X, save_dir=save_dir)
