from pathlib import Path

from runner.augmentation.cyclic_augmentation import CyclicAugment
from utils.data.dataset import VbsDataSetFactory

if __name__ == '__main__':
    save_dir = Path("/mnt/share/vbs-power-line-fault-detection/aug/")
    save_dir.mkdir(exist_ok=True, parents=True)
    augmentation = CyclicAugment(VbsDataSetFactory()(is_train=True), n_aug=20, save_dir=save_dir)
    augmentation.save_all_augs()
