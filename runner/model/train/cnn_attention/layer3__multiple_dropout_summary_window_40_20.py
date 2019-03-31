from pathlib import Path

from model.custom_nn import BasicCnnAttentionParams, CnnBlockParams
from runner.model.base_runner import BasicCnnAttentionTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/summary/aug/window_{}_step_{}/train".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/cnn/attention/layer_3_dilation_summary_window_{}_step_{}".format(
            40, 20))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VbsDataSetFactory()(is_train=True)
    params = BasicCnnAttentionParams(
        num_classes=1,
        cnn_blocks=[
            CnnBlockParams(
                in_feature=1,
                middle_feature=16,  # dummy
                out_feature=16,
                dropout_rate=0.4,  # dummy
                kernel_sizes=(10, 15),
                strides=(2, 1),
                padding=(0, 0),
                pool_kernel_size=3,
                pool_padding=1,
                pool_stride=2,
                concat_pool=False,
                dilation=1
            ),
            CnnBlockParams(
                in_feature=16,
                middle_feature=32,
                out_feature=32,
                dropout_rate=0.4,
                kernel_sizes=10,
                strides=5,
                padding=0,
                pool_kernel_size=0,
                pool_padding=1,
                pool_stride=4,
                # n_dropouts=4,
                concat_pool=False,
                dilation=1,
            ),
        ],
        last_chunk_len=398,
        context_size=16,
        last_dropout_rate=0.4,
        # last_n_dropouts=4
    )

    runner = BasicCnnAttentionTrainRunner(params,
                                          train_batch_size=160, valid_batch_size=1024,
                                          n_epochs=400, lr=2e-3, weight_decay=1e-5, patience=100,
                                          n_fold=5, aug_ratio=0)

    runner(train_dataset, saved_feature_root, save_dir=save_dir, validation_metric="loss")
