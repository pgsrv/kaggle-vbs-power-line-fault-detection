from pathlib import Path

from feature.base import SummaryWindowFeature
from model.custom_nn import CnnBlockMultipleDropoutParams, CnnAttentionMultipleDropoutParams
from runner.model.base_runner import CnnAttentionMultipleDropoutKfoldModel
from runner.model.test.custum_nn.base_test_runner import run

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/cnn/attention_multiple_dropout_glu/summary_window_{}_step_{}".format(
            40, 20))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(40, 20)
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

    run(feature, saved_feature_root, params, CnnAttentionMultipleDropoutKfoldModel, save_dir,
        batch_size=800, padded=False)

    # feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    # X = feature.feature_array
    # test_dataset = VbsDataSetFactory()(is_train=True)
    # test_dataset = TorchSimpleDataset(feature.feature_array, test_dataset.measurement_ids)
    #
    # df = model.predict(test_dataset, batch_size=1024, suffix="train")
    print("done!!")
