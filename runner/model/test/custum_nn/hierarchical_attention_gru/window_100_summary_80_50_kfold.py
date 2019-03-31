from pathlib import Path

import numpy as np

from feature.base import SummaryWindowFeature
from model.custom_nn import HierarchicalGruParams
from runner.model.base_runner import HierarchicalAttentionKfoldModel
from utils.data.dataset import VbsDataSetFactory, TorchSimpleDataset

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(400, 200))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_gru/summary/window_{}_step_{}".format(100, 200))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(400, 200)
    print("loading data")
    feature.load(list(saved_feature_root.glob("test/feature*.pickle")))
    X = feature.get_all_rows()

    X = np.pad(X, pad_width=((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0)

    print(X.shape)
    train_dataset = VbsDataSetFactory()(is_train=True)
    params = HierarchicalGruParams(
        seq_len=80,
        chunk_len=50,
        input_size=15,
        hidden_sizes=(64, 64),
        gru_dropout=0,
        dense_output=64,
        dropout=0.3,
        num_layers=1,
        pool_size=None,
        pool_stride=None
    )

    test_dataset = VbsDataSetFactory()(is_train=False)
    test_dataset = TorchSimpleDataset(X, test_dataset.measurement_ids)

    model = HierarchicalAttentionKfoldModel(params, model_path_root=save_dir, threshold=0.5)

    df = model.predict(test_dataset, batch_size=512)
    submission_dfs = model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(save_dir.joinpath("submission_{}.csv".format(key)), index=None)

    print("done!!")
