from pathlib import Path

from feature.base import SummaryWindowFeature
from model.custom_nn import HierarchicalGruParams
from runner.model.base_runner import HierarchicalAttentionKfoldModel
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_gru/summary/window_{}_step_{}".format(200, 200))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(40, 20)
    print("loading data")
    feature.load(list(sorted(saved_feature_root.glob("test/feature*.picklepadded.pickle"))))
    # X = feature.get_all_rows()
    #
    # print(X.shape)
    train_dataset = VbsDataSetFactory()(is_train=False)
    params = HierarchicalGruParams(
        seq_len=200,
        chunk_len=200,
        input_size=15,
        hidden_sizes=(128, 128),
        gru_dropout=0,
        dense_output=64,
        dropout=0.3,
        num_layers=1,
        pool_size=None,
        pool_stride=None
    )

    # test_dataset = VbsDataSetFactory()(is_train=False)
    # test_dataset = TorchSimpleDataset(X, test_dataset.measurement_ids)

    model = HierarchicalAttentionKfoldModel(params, model_path_root=save_dir, threshold=0.5)

    df = model.predict(feature, batch_size=128, measurement_ids=train_dataset.measurement_ids)
    submission_dfs = model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(save_dir.joinpath("submission_{}.csv".format(key)), index=None)

    print("done!!")
