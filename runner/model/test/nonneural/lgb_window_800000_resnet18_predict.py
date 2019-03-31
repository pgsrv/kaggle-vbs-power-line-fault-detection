from pathlib import Path

from feature.base import Features
from feature.plot_images import FrozenResnet18Feature
from model.lgb import LightGBMWrapper
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    feature_file = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/test/resnet18/window_800000_stride_80000/feature.csv")
    save_dir = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/lgb/window_800000_stride_800000_resnet18")
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = VbsDataSetFactory()(is_train=False)
    lgb_wrapper = LightGBMWrapper().load(save_dir.joinpath("model"))
    feature = Features([FrozenResnet18Feature]).load([feature_file])
    lgb_wrapper.predict(feature, vbs_dataset=dataset, save_dir=save_dir)
    print("done!")
