from pathlib import Path

from model.custom_nn import SENetParams, SEResNet1DBottleneck
from runner.model.base_runner import SENetPathTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/summary/aug/window_{}_step_{}/train".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/senet/resnet_50/aug_summary_window_{}_step_{}_ratio_4".format(
            40, 20))
    save_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = VbsDataSetFactory()(is_train=True)
    params = SENetParams(
        input_height=15,
        block=SEResNet1DBottleneck,
        layers=[3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=0.4,
        inplanes=64,
        downsample_kernel_size=3,
        downsample_padding=1,
        # last_pool_size = 14,
        block_stride=2,
        first_kernel_size=10,
        first_stride=5,
        n_first_conv=2,
    )

    runner = SENetPathTrainRunner(params,
                                  train_batch_size=192, valid_batch_size=192,
                                  n_epochs=200, lr=1e-3, weight_decay=1e-4, patience=40,
                                  n_fold=5, aug_ratio=4)

    runner(train_dataset, saved_feature_root, save_dir=save_dir, validation_metric="loss")
