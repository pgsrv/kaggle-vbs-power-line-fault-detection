from pathlib import Path

from feature.base import AverageWindowFeature
from model.custom_nn import SENetParams, SEResNet1DBottleneck
from runner.model.base_runner import SENetTrainRunner
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/avg/"
                              "window_{}_step_{}".format(8, 4))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/senet/resnet_50/avg_window_{}_step_{}".format(
            8, 4))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = AverageWindowFeature(8, 4)
    feature.load(saved_feature_root.joinpath("train/feature.pickle"))
    feature.feature_array = feature.feature_array.reshape((feature.feature_array.shape[0], 3, -1))
    X = feature.feature_array

    print(X.shape)
    train_dataset = VbsDataSetFactory()(is_train=True)
    params = SENetParams(
        input_height=3,
        block=SEResNet1DBottleneck,
        layers=[3, 4, 6, 3],
        groups=1,
        reduction=16,
        dropout_p=0.2,
        inplanes=32,
        downsample_kernel_size=3,
        downsample_padding=1,
        # last_pool_size = 14,
        block_stride=2,
        should_transpose=False
    )

    runner = SENetTrainRunner(params,
                              train_batch_size=12, valid_batch_size=4,
                              n_epochs=200, lr=2e-3, weight_decay=1e-5, patience=40,
                              n_fold=5)

    runner(train_dataset, X, save_dir=save_dir, num_workers=0, validation_metric="loss")

    # run(feature, saved_feature_root, params, SENetKfoldModel, save_dir, batch_size=80)