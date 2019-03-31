from pathlib import Path

from feature.base import WindowStandardScaled
from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionKfoldModel
from utils.data.dataset import VbsDataSetFactory, TorchSimpleDataset

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/fft/"
                              "/pca/fft_length_{}_stride_{}/normalized".format(5000, 2500))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/fft/fft_5000_pca_200_scaled")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = WindowStandardScaled(5000, 2500, accept_window=True)
    X = feature.load(list(sorted(saved_feature_root.glob("test/feature*.pickle")))).get_all_rows()
    print(X.shape)
    test_dataset = VbsDataSetFactory()(is_train=False)
    test_dataset = TorchSimpleDataset(X, test_dataset.measurement_ids)

    params = GruParams(
        seq_len=X.shape[1],
        input_size=X.shape[-1],
        hidden_size=64,
        gru_dropout=0.7,
        dense_output=32,
        dropout=0.7
    )

    model = BiGruAttensionKfoldModel(params, model_path_root=save_dir, threshold=0.5)

    df = model.predict(test_dataset, batch_size=1024)
    submission_dfs = model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(save_dir.joinpath("submission_{}.csv".format(key)), index=None)

    print("done!!")
