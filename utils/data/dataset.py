import dataclasses
import os
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.externals import joblib
from torch.utils.data import Dataset

RANDOM_STATE = 10


class VbsDataSetFactory(object):
    data_root = Path('../input/')
    if not data_root.exists():
        data_root = Path(__file__).parent.parent.parent.joinpath("data/raw")
    TRAIN_PARQUET_PATH = data_root.joinpath("train.parquet")
    TRAIN_METADATA_PATH = data_root.joinpath('metadata_train.csv')
    TEST_PARQUET_PATH = data_root.joinpath('test.parquet')
    TEST_MATADATA_PATH = data_root.joinpath('metadata_test.csv')
    SAMPLE_SUBMISSION_PATH = data_root.joinpath("sample_submission.csv")

    ID_MEASUREMENT_COLUMN = "id_measurement"
    SIGNAL_LENGTH = 800000

    def __init__(self):
        self.train_df = pd.read_csv(self.TRAIN_METADATA_PATH, encoding="utf-8")
        self.test_df = pd.read_csv(self.TEST_MATADATA_PATH, encoding="utf-8")
        self.submission_df = pd.read_csv(self.SAMPLE_SUBMISSION_PATH, encoding="utf-8")

    def __call__(self, is_train=True, *args, **kwargs):
        if is_train:
            return VbsDataSet(self.train_df, self.TRAIN_PARQUET_PATH)
        return VbsDataSet(self.test_df, self.TEST_PARQUET_PATH)


@dataclasses.dataclass
class VbsSignal:
    id: int
    parquet_path: Path


@dataclasses.dataclass
class VbsMeasurement:
    measurement_id: int
    parquet_path: Path
    signals: List[VbsSignal]

    def load_as_df(self):
        return pq.read_pandas(self.parquet_path, columns=[str(id) for id in self.get_signal_ids()]).to_pandas()

    def get_stacked_array(self):
        return self.load_as_df().values.transpose()

    def get_signal_ids(self):
        return [signal.id for signal in self.signals]


@dataclasses.dataclass
class VbsTrainMeasurement(VbsMeasurement):
    measurement_id: int
    parquet_path: Path
    signals: List[VbsSignal]
    target: int


@dataclasses.dataclass
class VbsDataSet:
    meta_df: pd.DataFrame
    parquet_path: Path

    def __post_init__(self):
        self.measurement_ids = list(sorted(self.meta_df[VbsDataSetFactory.ID_MEASUREMENT_COLUMN].unique().tolist()))
        self.is_train = False
        self.measurements: List[VbsMeasurement] = [self.get_signal_in_measurement(m_id) for m_id in self.measurement_ids
                                                   ]
        if "target" in self.meta_df.columns:
            self.targets = [measurement.target for measurement in self.measurements]

    def get_signal_in_measurement(self, measurement_id):
        df = self.meta_df[self.meta_df[VbsDataSetFactory.ID_MEASUREMENT_COLUMN] == measurement_id]

        if "target" in self.meta_df.columns:
            self.is_train = True
            return VbsTrainMeasurement(measurement_id, self.parquet_path,
                                       [VbsSignal(row[1]["signal_id"], self.parquet_path)
                                        for row in df.iterrows()], df["target"].values[0])
        else:
            return VbsMeasurement(measurement_id, self.parquet_path,
                                  [VbsSignal(row[1]["signal_id"], self.parquet_path)
                                   for row in df.iterrows()])

    def get_signals_in_phase(self, phase):
        df = self.meta_df[self.meta_df["phase"] == phase]
        df = pq.read_pandas(self.parquet_path, columns=[str(id) for id in df["signal_id"]]).to_pandas()
        return df.transpose()

    def get_flat_signals_in_phase(self, phase):
        return self.get_signals_in_phase(phase).values.reshape((-1, 1))

    def get_stacked_matrix(self):
        return np.stack([m.get_stacked_array() for m in self.measurements])

    def save_as_matrix(self, save_path):

        if len(self.measurements) > 3000:
            split_size = 3000 // 2
            n_split = len(self.measurement_ids) // split_size
            n_split += int(len(self.measurement_ids) % split_size > 0)
            for i in range(n_split):
                start = i * split_size
                end = (i + 1) * split_size
                if end < len(self.measurement_ids):
                    x = np.stack([m.get_stacked_array() for m in self.measurements[start:end]])
                else:
                    x = np.stack([m.get_stacked_array() for m in self.measurements[start:]])
                joblib.dump(x, str(save_path.joinpath("feature_{}.pickle".format(i))))
                return
        raise ValueError("not implemented")


n_cpus = os.cpu_count()


class TorchDelayedRawDataset(Dataset):

    def __init__(self, vbsdataset: VbsDataSet, measurement_ids):
        self.raw_dataset = vbsdataset
        self.measurement_ids = measurement_ids

    def __len__(self):
        return len(self.measurement_ids)

    def __getitem__(self, index):
        measurement = self.raw_dataset.get_signal_in_measurement(self.measurement_ids[index])
        X = measurement.get_stacked_array().astype("float32")
        if isinstance(measurement, VbsTrainMeasurement):
            return {"image": torch.from_numpy(X), "label": torch.Tensor([measurement.target])}
        return {"image": torch.from_numpy(X)}


class TorchSimpleDataset(Dataset):

    def __init__(self, X, measure_ids, y=None, is_eval=False):
        self.is_eval = is_eval
        self.X = X.astype("float32")
        self.measurement_ids = measure_ids
        if y is not None:
            self.Y = y.astype("float32")
        else:
            self.Y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if self.Y is None:
            return {"image": self._to_tensor(self.X[index])}
        return {"image": self._to_tensor(self.X[index]), "label": self._to_tensor([self.Y[index]])}

    def _to_tensor(self, x):
        if self.is_eval:
            return torch.tensor(x, requires_grad=False)
        return torch.tensor(x)


class TorchMixUpDataset(Dataset):

    def __init__(self, X, measure_ids, y, mixup_ratio=4, alpha=2, beta=2, oversampling=False):
        self.beta = beta
        self.alpha = alpha
        self.X = X.astype("float32")
        self.Y = y
        self.measurement_ids = measure_ids
        self.mixup_size = int(self.X.shape[0] * mixup_ratio)
        self.oversampling = oversampling
        self.prepare_mix_up()

    def prepare_mix_up(self):
        np.random.seed(RANDOM_STATE)
        if self.oversampling:
            sampler = RandomOverSampler(RANDOM_STATE)
            src_indices, _ = sampler.fit_resample(range(self.X.shape[0]), y=self.Y)
        else:
            src_indices = np.asarray(range(0, self.X.shape[0]))
        self._set_indices = np.random.choice(src_indices, size=(self.mixup_size, 2), replace=True)
        self._lambdas = np.random.beta(self.alpha, self.beta, self.mixup_size)

    def __len__(self):
        return self.X.shape[0] + self.mixup_size

    def __getitem__(self, index):
        # if self.Y is None:
        #     return {"image": torch.from_numpy(self.X[index])}
        if index >= self.X.shape[0]:
            mixup_idx = index - self.X.shape[0]
            idxs = self._set_indices[mixup_idx]
            lambda_ = self._lambdas[mixup_idx]

            x = lambda_ * self.X[idxs[0]] + (1 - lambda_) * self.X[idxs[1]]
            y = int((lambda_ * self.Y[idxs[0]] + (1 - lambda_) * self.Y[idxs[1]]) > 0.5)

        else:
            x = self.X[index]
            y = self.Y[index]
        return {"image": torch.from_numpy(x), "label": torch.Tensor([y])}


class CyclicAugmentationDataset(TorchSimpleDataset):

    def __init__(self, X, measure_ids, y, aug_ratio=4, oversampling=False):
        super().__init__(X, measure_ids, y)
        self.aug_size = int(self.X.shape[0] * aug_ratio)
        self.oversampling = oversampling
        self.prepare_aug()

    def prepare_aug(self):
        np.random.seed(RANDOM_STATE)
        if self.oversampling:
            sampler = RandomOverSampler(RANDOM_STATE)
            src_indices, _ = sampler.fit_resample(range(self.X.shape[0]), y=self.Y)
        else:
            src_indices = np.asarray(range(0, self.X.shape[0]))
        self._set_indices = np.random.choice(src_indices, size=self.aug_size, replace=False)
        self._split_points = np.random.uniform(1, self.src_len() - 1, size=self.aug_size)

    def __len__(self):
        return self.X.shape[0] + self.aug_size

    def src_len(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if index >= self.X.shape[0]:
            aug_idx = index - self.X.shape[0]
            augmented_idx = self._set_indices[aug_idx]
            point = self._split_points[aug_idx]
            src_x = self.X[augmented_idx]
            x = np.concatenate([src_x[:, :, point:], src_x[:, :, :point]], axis=2)
            y = self.Y[augmented_idx]
        else:
            x = self.X[index]
            y = self.Y[index]
        return {"image": self._to_tensor(x), "label": self._to_tensor([y])}


class AugmentedFeaturePathDataset(Dataset):

    def __init__(self, aug_root: Path, measure_ids, y, aug_ratio=4, oversampling=False, is_eval=False):
        self.raw_paths = OrderedDict([(i, aug_root.glob("{}/raw*.pickle".format(i))) for i in measure_ids])

        self.augmented_paths = OrderedDict([(i, list(aug_root.glob("{}/aug*.pickle".format(i))))
                                            for i in measure_ids])
        self.aug_root = aug_root

        assert len(measure_ids) == len(y)
        self.aug_ratio = aug_ratio
        self.aug_size = int(self.aug_ratio * len(measure_ids))
        self.measure_ids = measure_ids
        self.y = y.astype("float32")
        self.y_dict = {measure_id: self.y[i:i + 1] for i, measure_id in enumerate(measure_ids)}
        self.is_eval = is_eval

        self.oversampling = oversampling if oversampling else 0
        if aug_ratio > 0:
            self.prepare_aug()

    def prepare_aug(self):
        np.random.seed(RANDOM_STATE)

        sampled_measure_ids = np.asarray(self.measure_ids)

        # if self.oversampling:
        #     # if (isinstance(self.oversampling, int) or isinstance(self.oversampling, float)) and self.oversampling > 1:
        #     #     measure_id_array = np.asarray(self.measure_ids).reshape((-1, 1))
        #     #     minority = measure_id_array[self.y == 1]
        #     #     sampled_measure_ids = resample(minority,
        #     #                                    n_samples=int(minority.shape[0] * self.oversampling),
        #     #                                    random_state=RANDOM_STATE)
        #     #     sampled_measure_ids = np.vstack([measure_id_array, sampled_measure_ids])
        #     # else:
        #     #     sampler = RandomOverSampler(random_state=RANDOM_STATE,
        #     #                                 sampling_strategy=self.oversampling)
        #     #     sampled_measure_ids, _ = sampler.fit_resample(
        #     #         np.asarray(self.measure_ids).reshape((-1, 1)),
        #     #         y=self.y)
        #     measure_id_array = np.asarray(self.measure_ids).reshape((-1, 1))
        minority_indices = (self.y == 1).astype('float32')

        self.weight = np.ones(self.y.shape) + minority_indices * self.oversampling
        self.weight = self.weight / np.sum(self.weight)

        sampled_measure_ids = np.asarray(sampled_measure_ids).reshape((-1))
        if self.aug_ratio > 1:
            self._augmented_indices = np.random.choice(sampled_measure_ids, size=self.aug_size, replace=True,
                                                       p=self.weight)
        else:
            self._augmented_indices = np.random.choice(sampled_measure_ids, size=self.aug_size, replace=False,
                                                       p=self.weight)
        count_table = np.unique(self._augmented_indices, return_counts=True)

        self._sampled_path_seeds = OrderedDict([(measure_id, self.augmented_paths[measure_id][:n_choice])
                                                for measure_id, n_choice
                                                in zip(count_table[0], count_table[1])])
        self._measure_id_used_count = OrderedDict([(measure_id, 0) for measure_id in count_table[0]])
        self._sampled_paths = [self._get_path_and_count(index) for index in self._augmented_indices]

    def _get_path_and_count(self, measure_id):
        path = self._sampled_path_seeds[measure_id]
        path = path[self._measure_id_used_count[measure_id]]
        self._measure_id_used_count[measure_id] += 1
        return path

    def __len__(self):
        return self.y.shape[0] + self.aug_size

    def src_len(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if self.aug_ratio > 0 and index >= self.src_len():
            augmented_idx = self._augmented_indices[index - self.src_len()]
            x = joblib.load(self._get_from_augmented_arrays(index - self.src_len()))
            y = self.y_dict[augmented_idx]
        else:
            x = joblib.load(self.aug_root.joinpath("{}/raw.pickle".format(self.measure_ids[index])))
            y = self.y[index: index + 1]
        return {"image": self._to_tensor(x.astype("float32")), "label": self._to_tensor(y)}

    def _get_from_augmented_arrays(self, index):
        return self._sampled_paths[index]

    def _to_tensor(self, x):
        if self.is_eval:
            return torch.tensor(x, requires_grad=False)
        else:
            return torch.tensor(x)
