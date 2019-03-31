from pathlib import Path

from model.custom_nn import HierarchicalGruParams
from runner.model.base_runner import HierarchicalRawAttentionTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_attention/raw/800_1000")
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VbsDataSetFactory()(is_train=True)
    params = HierarchicalGruParams(
        seq_len=400,
        chunk_len=400,
        input_size=3,
        hidden_sizes=(128, 128),
        gru_dropout=0.1,
        dense_output=64,
        dropout=0.3,
        num_layers=1,
        pool_size=100,
        pool_stride=50
    )

    runner = HierarchicalRawAttentionTrainRunner(params,
                                                 train_batch_size=1, valid_batch_size=1,
                                                 n_epochs=100, lr=2e-3, weight_decay=1e-5, patience=20,
                                                 n_fold=10)

    runner(save_dir=save_dir, is_train=True, num_workers=None)
