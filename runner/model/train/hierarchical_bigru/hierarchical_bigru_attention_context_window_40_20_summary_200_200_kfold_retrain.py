from pathlib import Path

from feature.base import SummaryWindowFeature
from model.custom_nn import HierarchicalGruParams
from runner.model.base_runner import HierarchicalAttentionTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_gru_context/summary/window_{}_step_{}_10fld".format(
            200, 200))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(40, 20)
    feature.load(saved_feature_root.joinpath("train/feature_padded.pickle"))
    X = feature.feature_array

    # X = np.pad(X, pad_width=((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=0)

    print(X.shape)
    train_dataset = VbsDataSetFactory()(is_train=True)
    params = HierarchicalGruParams(
        seq_len=200,
        chunk_len=200,
        input_size=15,
        hidden_sizes=(128, 128),
        gru_dropout=0,
        dense_output=64,
        dropout=0.4,
        num_layers=1,
        pool_size=None,
        pool_stride=None,
        attention_type="context",
        context_size=(64, 64),
        use_hidden=True
    )

    runner = HierarchicalAttentionTrainRunner(params,
                                              train_batch_size=48, valid_batch_size=48,
                                              n_epochs=400, lr=2e-3, weight_decay=1e-5, patience=100,
                                              n_fold=5)

    runner(train_dataset, X, save_dir=save_dir, num_workers=0, validation_metric="loss")
