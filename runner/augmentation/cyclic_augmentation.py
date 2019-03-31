from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
from sklearn.externals import joblib

from utils.data.dataset import VbsDataSet, VbsDataSetFactory, n_cpus


class CyclicAugment(object):
    def __init__(self, dataset: VbsDataSet, n_aug, save_dir: Path, n_jobs=None):
        self.dataset = dataset
        self.n_aug = n_aug
        self.seq_len = VbsDataSetFactory.SIGNAL_LENGTH
        self.save_dir = save_dir
        if n_jobs is None:
            n_jobs = n_cpus
        self.n_jobs = n_jobs

    def save_all_augs(self):
        with Pool(self.n_jobs) as pool:
            rows = pool.map(self.write_augs, self.dataset.measurement_ids)

    def write_augs(self, i):
        print("writing {} / {} ".format(i, self.dataset.meta_df.shape[0]))
        sample_root = self.save_dir.joinpath(str(i))
        sample_root.mkdir(exist_ok=True, parents=True)
        x = self.dataset.measurements[i].get_stacked_array()

        joblib.dump(x, str(sample_root.joinpath("raw.pickle")))

        split_points = np.random.randint(1, self.seq_len - 1, size=self.n_aug)
        for j, point in enumerate(split_points):
            new_x = np.concatenate([x[:, point:], x[:, :point]], axis=1)
            joblib.dump(new_x, str(sample_root.joinpath("aug_{}_point_{}.pickle".format(j, point))))

    #
    # self._set_indices = np.random.choice(src_indices, size=self.aug_size, replace=False)
    #
    #
    # def prepare_aug(self):
    #     # np.random.seed(RANDOM_STATE)
    #     if self.oversampling:
    #         sampler = RandomOverSampler(RANDOM_STATE)
    #         src_indices, _ = sampler.fit_resample(range(self.X.shape[0]), y=self.Y)
    #     else:
