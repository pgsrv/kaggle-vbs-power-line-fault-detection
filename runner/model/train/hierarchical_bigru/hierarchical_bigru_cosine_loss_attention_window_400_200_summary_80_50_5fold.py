from pathlib import Path

import numpy as np

from feature.base import SummaryWindowFeature
from model.custom_nn import HierarchicalGruParams
from runner.model.base_runner import HierarchicalAttentionTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(400, 200))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_gru/cosine_loss/summary/window_{}_step_{}".format(
            100, 200))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(400, 200)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    X = feature.feature_array

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

    runner = HierarchicalAttentionTrainRunner(params,
                                              train_batch_size=128, valid_batch_size=512,
                                              n_epochs=300, lr=1e-3, weight_decay=1e-5, patience=50,
                                              n_fold=5, loss="cosine")

    runner(train_dataset, X, save_dir=save_dir, validation_metric="loss")
