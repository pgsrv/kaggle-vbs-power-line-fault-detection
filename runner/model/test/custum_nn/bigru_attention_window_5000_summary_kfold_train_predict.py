from pathlib import Path

from feature.base import PercentileSummaryWindowFeature
from model.custom_nn import GruParams
from runner.model.base_runner import BiGruAttensionKfoldModel
from utils.data.dataset import VbsDataSetFactory, TorchSimpleDataset

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/percentile_summary/"
                              "window_{}_step_{}".format(5000, 5000))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/bigru_attention/window_5000_summary")
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = PercentileSummaryWindowFeature(5000, 5000)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array
    test_dataset = VbsDataSetFactory()(is_train=True)
    test_dataset = TorchSimpleDataset(feature.feature_array, test_dataset.measurement_ids)
    params = GruParams(
        seq_len=VbsDataSetFactory.SIGNAL_LENGTH // 5000,
        input_size=X.shape[-1],
        hidden_size=96,
        gru_dropout=0.3,
        dense_output=64,
        dropout=0.3
    )

    model = BiGruAttensionKfoldModel(params, model_path_root=save_dir, threshold=0.5)

    df = model.predict(test_dataset, batch_size=512, suffix="train")
    # submission_dfs = model.to_submission(df)
    # for key, submission_df in submission_dfs.items():
    #     submission_df.to_csv(save_dir.joinpath("train_submission_{}.csv".format(key)), index=None)

    print("done!!")
