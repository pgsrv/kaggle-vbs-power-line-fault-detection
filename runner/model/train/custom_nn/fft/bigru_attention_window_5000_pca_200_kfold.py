from pathlib import Path

from feature.base import PcaFeature
from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/fft/"
                              "/pca/fft_length_{}_stride_{}".format(5000, 2500))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/fft/fft_5000_pca_200")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = PcaFeature(200)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array
    train_dataset = VbsDataSetFactory()()

    params = GruParams(
        seq_len=X.shape[1],
        input_size=X.shape[-1],
        hidden_size=96,
        gru_dropout=0.3,
        dense_output=64,
        dropout=0.3
    )
    runner = BiGruAttensionTrainRunner(params,
                                       train_batch_size=256, valid_batch_size=256,
                                       n_epochs=200, lr=2e-3, weight_decay=1e-5, patience=100,
                                       n_fold=10)

    runner(train_dataset, X, save_dir=save_dir)
