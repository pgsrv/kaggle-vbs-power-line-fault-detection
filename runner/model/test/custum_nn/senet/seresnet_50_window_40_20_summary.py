from pathlib import Path

from feature.base import SummaryWindowFeature
from model.custom_nn import SENetParams, SEResNet1DBottleneck
from runner.model.base_runner import SENetKfoldModel
from runner.model.test.custum_nn.base_test_runner import run

if __name__ == '__main__':
    saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
                              "window_{}_step_{}".format(40, 20))

    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/senet/resnet_10/summary_window_{}_step_{}".format(
            40, 20))
    save_dir.mkdir(parents=True, exist_ok=True)

    feature = SummaryWindowFeature(40, 20)

    params = SENetParams(
        input_height=15,
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
    )

    run(feature, saved_feature_root, params, SENetKfoldModel, save_dir, batch_size=80, padded=True)
