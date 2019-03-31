from pathlib import Path

from model.custom_nn import CnnAttentionMultipleDropoutParams, \
    CnnBlockMultipleDropoutParams
from runner.model.base_runner import CnnAttentionMultipleDropoutTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/summary/aug/window_{}_step_{}/train".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/cnn/attention_multiple_dropout/summary_window_{}_step_{}".format(
            40, 20))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VbsDataSetFactory()(is_train=True)
    params = CnnAttentionMultipleDropoutParams(
        num_classes=1,
        cnn_blocks=[
            # seq len 39999
            CnnBlockMultipleDropoutParams(
                in_feature=1,
                middle_feature=16,  # dummy
                out_feature=16,
                dropout_rate=0.3,  # dummy
                kernel_sizes=(3, 15),
                strides=(2, 1),
                padding=(1, 0),
                pool_kernel_size=3,
                pool_padding=1,
                pool_stride=2,
                # dilation: int = 1,
                # n_dropouts=5
            ),
            # seq len 10000
            CnnBlockMultipleDropoutParams(
                in_feature=32,
                middle_feature=32,
                out_feature=32,
                dropout_rate=0.3,
                kernel_sizes=3,
                strides=2,
                padding=1,
                pool_kernel_size=3,
                pool_padding=1,
                pool_stride=2,
                # dilation: int = 1,
                n_dropouts=4
            ),
            # seq len
            CnnBlockMultipleDropoutParams(
                in_feature=64,
                middle_feature=64,
                out_feature=64,
                dropout_rate=0.3,
                kernel_sizes=3,
                strides=2,
                padding=1,
                pool_kernel_size=0,
                pool_padding=0,
                pool_stride=2,
                # dilation: int = 1,
                n_dropouts=4
            ),
        ],
        last_chunk_len=313,
        context_size=32,
        last_dropout_rate=0.4,
        last_n_dropouts=4
    )

    runner = CnnAttentionMultipleDropoutTrainRunner(params,
                                                    train_batch_size=128, valid_batch_size=800,
                                                    n_epochs=400, lr=2e-3, weight_decay=1e-5, patience=100,
                                                    n_fold=5, aug_ratio=0)

    runner(train_dataset, saved_feature_root, save_dir=save_dir, validation_metric="loss")
