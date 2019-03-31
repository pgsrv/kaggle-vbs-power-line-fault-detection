from pathlib import Path

from feature.base import WindowStandardScaled
from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionMixupTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/fft/"
                              "/pca/fft_length_{}_stride_{}/normalized".format(5000, 2500))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/fft/fft_5000_pca_200_scaled_mixup_3")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = WindowStandardScaled(5000, 2500, accept_window=True)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array
    train_dataset = VbsDataSetFactory()()

    params = GruParams(
        seq_len=X.shape[1],
        input_size=X.shape[-1],
        hidden_size=64,
        gru_dropout=0.7,
        dense_output=32,
        dropout=0.7
    )
    runner = BiGruAttensionMixupTrainRunner(params,
                                            train_batch_size=256, valid_batch_size=256,
                                            n_epochs=100, lr=2e-3, weight_decay=1e-4, patience=50,
                                            n_fold=10, mixup_ratio=3)

    runner(train_dataset, X, save_dir=save_dir)
