# #====================== for debug===============
# sys.path.append(str(Path(__file__).parent.parent.parent.parent.joinpath("pycharm-debug-py3k.egg")))
# import pydevd
#
# pydevd.settrace('local-dev', port=12345, stdoutToServer=True,
# stderrToServer=True)
# #===============================================
import logging
import math
import random
import re
import sys
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from imblearn import under_sampling
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import binarize
from tqdm import tqdm

from feature.base import RANDOM_STATE, Feature
from model.nn_model_wrapper import NnModelWrapper, \
    BiGruAttension, BiLstmAttension, Gru, HierarchicalAttention, SENetWrapper, CnnAttentionMultipleDropoutWrapper, \
    BasicCnnAttentionWrapper
from utils.data.dataset import VbsDataSetFactory, n_cpus, VbsDataSet, TorchSimpleDataset, TorchMixUpDataset, \
    TorchDelayedRawDataset, AugmentedFeaturePathDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


class TrainRunner(object, metaclass=ABCMeta):
    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10,
                 lr=1e-4, weight_decay=1e-5, is_debug=False, valid_size=0.2, loss=None):
        self.patience = patience
        self.n_epochs = n_epochs
        self.valid_batch_size = valid_batch_size
        self.train_batch_size = train_batch_size
        self.dropout_rate = dropout_rate
        self.is_debug = is_debug
        self.lr = lr
        self.weight_decay = weight_decay
        self.valid_size = valid_size
        self.loss = loss

    def __call__(self, dataset: VbsDataSet, train_feature_matrix, save_dir):
        pass
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
        # train_Y = dataset.meta_df["target"].values
        #
        # train_feature_matrix, train_Y, train_measurement_ids,
        # transformers = ImagenetTransformers()
        # train_dataset = PlotImageDataset(train_plot_meta_df, plot_root_path, transformers=transformers)
        # valid_dataset = PlotImageDataset(valid_plot_meta_df, plot_root_path, transformers=transformers)
        # # save_dir = Path("/mnt/gcs/kaggle-grasp-and-lift-eeg-detection/model/vgg_pretrained/window_750_stride_75")
        # save_dir.mkdir(exist_ok=True, parents=True)
        #
        # file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        # file_handler.setLevel(logging.INFO)
        # logger.addHandler(file_handler)
        #
        # model_wrapper = self.create_model(save_dir)
        #
        # if self.is_debug:
        #     DEBUG_SIZE = 4000
        #     train_dataset.train_dataset.label_df = train_dataset.train_dataset.label_df[:DEBUG_SIZE]
        #     train_dataset.train_dataset.plot_paths = train_dataset.train_dataset.plot_paths[:DEBUG_SIZE]
        #     train_dataset.valid_dataset.label_df = train_dataset.valid_dataset.label_df[:DEBUG_SIZE]
        #     train_dataset.valid_dataset.plot_paths = train_dataset.valid_dataset.plot_paths[:DEBUG_SIZE]
        # model_wrapper.train(train_dataset, valid_dataset,
        #                     train_batch_size=self.train_batch_size, valid_batch_size=self.valid_batch_size,
        #                     n_epochs=self.n_epochs, patience=self.patience,
        #                     num_workers=n_cpus)

    @staticmethod
    def train_valid_split_df(df, valid_size):
        n_rows = df.shape[0]
        random.seed(RANDOM_STATE)
        n_valid_samples = math.floor(n_rows * valid_size)
        shuffled_indices = list(range(n_rows))
        random.shuffle(shuffled_indices)
        valid_indices = shuffled_indices[:n_valid_samples]
        train_indices = shuffled_indices[n_valid_samples:]
        return df.iloc[train_indices, :].reset_index(), df.iloc[valid_indices, :].reset_index()

    @abstractmethod
    def create_model(self, save_dir):
        pass


class KfoldTrainRunner(TrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, loss=None):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, loss=loss)
        self.n_fold = n_fold

    def __call__(self, dataset: VbsDataSet, train_feature_matrix, save_dir, num_workers=None,
                 validation_metric="score"):
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
        self.num_workers = num_workers
        if isinstance(train_feature_matrix, np.ndarray):
            self.feature_matrix = train_feature_matrix.astype("float32")
        else:
            self.feature_matrix = np.asarray(train_feature_matrix)
        self.Y = np.asarray(dataset.targets)
        self.measurement_ids = dataset.measurement_ids
        self.raw_dataset = dataset
        self.validation_metric = validation_metric

        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir

        self.add_logger(save_dir)

        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=RANDOM_STATE) \
            .split(self.feature_matrix, self.Y)

        for i, (train_index, test_index) in enumerate(folds):
            logger.info("****cv {} / {} ****".format(i, self.n_fold))
            self._train_cv(i, train_index, test_index)

    def add_logger(self, save_dir):
        file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    def _train_cv(self, i, train_index, test_index):
        train_X = self.feature_matrix[train_index]
        valid_X = self.feature_matrix[test_index]
        train_Y = self.Y[train_index]
        valid_Y = self.Y[test_index]
        train_measurement_ids = [self.measurement_ids[idx] for idx in train_index]
        valid_measurement_ids = [self.measurement_ids[idx] for idx in test_index]
        cv_root = self.save_dir.joinpath("cv{}".format(i))
        cv_root.mkdir(exist_ok=True, parents=True)
        self.write_cv(cv_root, train_measurement_ids, valid_measurement_ids)

        train_dataset, valid_dataset = self.create_dataset(train_X, train_Y, train_measurement_ids, valid_X, valid_Y,
                                                           valid_measurement_ids)

        model_wrapper = self.create_model(cv_root)

        if self.num_workers is None:
            self.num_workers = n_cpus

        model_wrapper.train(train_dataset, valid_dataset,
                            train_batch_size=self.train_batch_size, valid_batch_size=self.valid_batch_size,
                            n_epochs=self.n_epochs, patience=self.patience,
                            num_workers=self.num_workers, validation_metric=self.validation_metric)

    def write_cv(self, cv_root, train_measurement_ids, valid_measurement_ids):
        with cv_root.joinpath("train.csv").open("w+") as f:
            f.write("\n".join([str(idx) for idx in train_measurement_ids]))
        with cv_root.joinpath("valid_csv").open("w+") as f:
            f.write("\n".join([str(idx) for idx in valid_measurement_ids]))

    def create_dataset(self, train_X, train_Y, train_measurement_ids, valid_X, valid_Y, valid_measurement_ids):
        train_dataset = TorchSimpleDataset(train_X, train_measurement_ids, train_Y)
        valid_dataset = TorchSimpleDataset(valid_X, valid_measurement_ids, valid_Y, is_eval=True)
        return train_dataset, valid_dataset

    @abstractmethod
    def create_model(self, save_dir):
        pass


class MixupKfoldTrainRunner(KfoldTrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, mixup_ratio=3):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold)
        self.mixup_ratio = mixup_ratio

    def create_dataset(self, train_X, train_Y, train_measurement_ids, valid_X, valid_Y, valid_measurement_ids):
        train_dataset = TorchMixUpDataset(train_X, train_measurement_ids, train_Y)
        valid_dataset = TorchSimpleDataset(valid_X, valid_measurement_ids, valid_Y)
        return train_dataset, valid_dataset


class PathKfoldTrainRunner(KfoldTrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, aug_ratio=4, oversampling=False,
                 loss=None):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, loss)
        self.aug_ratio = aug_ratio
        self.oversampling = oversampling

    def __call__(self, dataset: VbsDataSet, feature_root, save_dir, num_workers=None,
                 validation_metric="score"):
        self.feature_root = feature_root
        dummy_x = dataset.measurement_ids
        super().__call__(dataset, dummy_x, save_dir, num_workers, validation_metric)

    def create_dataset(self, train_X, train_Y, train_measurement_ids, valid_X, valid_Y, valid_measurement_ids):
        train_dataset = AugmentedFeaturePathDataset(self.feature_root, train_measurement_ids, train_Y,
                                                    aug_ratio=self.aug_ratio,
                                                    oversampling=self.oversampling)
        valid_dataset = AugmentedFeaturePathDataset(self.feature_root, valid_measurement_ids, valid_Y,
                                                    aug_ratio=0, is_eval=True)
        return train_dataset, valid_dataset


class UndersampleBaggingKfoldTrainRunner(TrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, n_bag=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size)
        self.n_bag = n_bag
        self.n_fold = n_fold

    def __call__(self, dataset: VbsDataSet, train_feature_matrix, save_dir, num_workers=None):
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
        self.num_workers = num_workers
        self.feature_matrix = train_feature_matrix.astype("float32")
        self.Y = np.asarray(dataset.targets)
        self.measurement_ids = dataset.measurement_ids
        self.raw_dataset = dataset

        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir

        file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=RANDOM_STATE).split(
            self.feature_matrix, self.Y)

        for i, (train_index, test_index) in enumerate(folds):
            logger.info("****cv {} / {} ****".format(i, self.n_fold))
            self._train_cv(i, train_index, test_index)

    def _train_cv(self, i, train_index, test_index):
        self._train_X = self.feature_matrix[train_index]
        self._valid_X = self.feature_matrix[test_index]
        self._train_Y = self.Y[train_index]
        self._valid_Y = self.Y[test_index]
        self.train_measurement_ids = [self.measurement_ids[idx] for idx in train_index]
        self.valid_measurement_ids = [self.measurement_ids[idx] for idx in test_index]
        cv_root = self.save_dir.joinpath("cv{}".format(i))
        cv_root.mkdir(exist_ok=True, parents=True)
        with cv_root.joinpath("train.csv").open("w+") as f:
            f.write("\n".join([str(idx) for idx in self.train_measurement_ids]))
        with cv_root.joinpath("valid_csv").open("w+") as f:
            f.write("\n".join([str(idx) for idx in self.valid_measurement_ids]))

        for j in range(self.n_bag):
            logger.info("---- training bag {} / {} ---- ".format(str(j), self.n_bag))
            bag_root = cv_root.joinpath("{}".format(j))
            bag_root.mkdir(exist_ok=True, parents=True)
            self._train_bag(j, bag_root)

    def _train_bag(self, random_state, bag_root: Path):
        indices, X, Y = self._get_bag(random_state)
        # df.to_csv(bag_root.joinpath("bag.csv"))

        model_wrapper = self.create_model(bag_root)

        train_dataset = TorchSimpleDataset(X, indices, Y)
        valid_dataset = TorchSimpleDataset(self._valid_X, self.valid_measurement_ids, self._valid_Y)

        # if self.is_debug:
        #     DEBUG_SIZE = 4000
        #     train_dataset.train_dataset.label_df = train_dataset.train_dataset.label_df[:DEBUG_SIZE]
        #     train_dataset.train_dataset.plot_paths = train_dataset.train_dataset.plot_paths[:DEBUG_SIZE]
        #     train_dataset.valid_dataset.label_df = train_dataset.valid_dataset.label_df[:DEBUG_SIZE]
        #     train_dataset.valid_dataset.plot_paths = train_dataset.valid_dataset.plot_paths[:DEBUG_SIZE]
        model_wrapper.train(train_dataset, valid_dataset,
                            train_batch_size=self.train_batch_size, valid_batch_size=self.valid_batch_size,
                            n_epochs=self.n_epochs, patience=self.patience,
                            num_workers=n_cpus)

    def _get_bag(self, random_state):
        sampler = under_sampling.RandomUnderSampler(sampling_strategy="majority", return_indices=False,
                                                    random_state=random_state, replacement=False)
        indices, _ = sampler.fit_sample(np.arange(self._train_X.shape[0]).reshape((-1, 1)),
                                        self._train_Y.reshape((-1, 1)))
        return indices, self._train_X[indices.reshape((-1)).tolist()], self._train_Y[indices.reshape((-1)).tolist()]

    @abstractmethod
    def create_model(self, save_dir):
        pass


class EnsembleModel(object, metaclass=ABCMeta):

    def __init__(self, threshold):
        self.threshold = threshold

    def _hard_vote(self, dfs: List[pd.DataFrame], class_col="class"):
        output_classes = np.hstack([df[class_col].values.reshape((-1, 1)) for df in dfs])
        n_majority = output_classes.shape[1] // 2 + int(output_classes.shape[1] % 2 > 0)
        voted = output_classes.sum(axis=1)
        return np.where(voted >= n_majority, 1, 0)

    def _soft_vote(self, dfs: List[pd.DataFrame], probability_col="probability"):
        avg_probability = np.hstack([df[probability_col].values.reshape((-1, 1)) for df in dfs]).mean(axis=1).reshape(
            (-1, 1))
        return avg_probability, binarize(avg_probability, threshold=self.threshold)


class KfoldModel(EnsembleModel, metaclass=ABCMeta):

    def __init__(self, model_path_root: Path, threshold=0.5):
        super().__init__(threshold)

        self.model_path_root = model_path_root

        self._current_fold = 0

    def load_model(self):
        self._model: NnModelWrapper = self.create_model(
            self.model_path_root.joinpath("cv{}/model".format(self._current_fold)))

    def predict(self, dataset: [TorchSimpleDataset, Feature], batch_size=256, suffix="", measurement_ids=None):
        folds = [int(path.name.replace("cv", "")) for path in self.model_path_root.glob("cv*")]
        self._fold_results = []
        self.suffix = suffix
        if measurement_ids is not None:
            self._measure_ids = measurement_ids
        else:
            self._measure_ids = dataset.measurement_ids

        for fold in folds:
            self._current_fold = fold
            self._predict_fold(dataset, batch_size)

        self._fold_df = pd.DataFrame()
        self._fold_df["id_measurement"] = self._measure_ids
        self._fold_df["hard_voted_class"] = self._hard_vote(self._fold_results)
        self._fold_df["avg_probability"], self._fold_df["soft_voted_class"] = self._soft_vote(self._fold_results)
        self._fold_df.to_csv(self.model_path_root.joinpath(suffix + "predicted.csv"))
        self._fold_results.append(self._fold_df)

        self._fold_df.to_csv(self.model_path_root.joinpath(suffix + "predicted.csv"))
        return self._fold_df

    def _predict_fold(self, dataset: [TorchSimpleDataset, Feature], batch_size=256):
        self._current_fold_root = self.model_path_root.joinpath("cv{}".format(self._current_fold))

        logger.info("**** predicting with fold {} ****".format(self._current_fold))
        self.load_model()
        if isinstance(dataset, Feature):
            predicted = np.concatenate(
                [self._model.predict(TorchSimpleDataset(x, self._measure_ids), batch_size, n_cpus)
                 for x in dataset.partial_load()], axis=0)
            print(predicted.shape)
            dataset.flash()
        else:
            predicted = self._model.predict(dataset, batch_size, n_cpus)
        df = pd.DataFrame()
        df["id_measurement"] = self._measure_ids
        print(df.shape)
        df["probability"] = predicted
        df["class"] = binarize(predicted, threshold=self.threshold)
        df.to_csv(self._current_fold_root.joinpath(self.suffix + "predicted.csv"))
        self._fold_results.append(df)

    @abstractmethod
    def create_model(self, model_path):
        pass

    @staticmethod
    def to_submission(df):
        logger.info("creating submission form")
        test_df = VbsDataSetFactory().test_df
        soft_df = df[["id_measurement", "soft_voted_class"]]
        hard_df = df[["id_measurement", "hard_voted_class"]]

        def f(df: pd.DataFrame):
            submission_df = test_df.merge(df, on="id_measurement", how="left")
            submission_df.drop(["id_measurement", "phase"], axis=1, inplace=True)
            col = [str(col) for col in submission_df.columns if str(col) != "signal_id"][0]
            submission_df.rename(columns={col: "target"}, inplace=True)
            submission_df["target"] = submission_df["target"].astype(bool)
            return submission_df

        submission_dfs = {}
        submission_dfs["soft"] = f(soft_df)
        submission_dfs["hard"] = f(hard_df)

        return submission_dfs


class BaggingKfoldModel(EnsembleModel, metaclass=ABCMeta):
    BAG_ROOT_PATTERN = re.compile(r"\d")

    def __init__(self, model_path_root: Path, dropout_rate=0.5, threshold=0.5):
        super().__init__(threshold)
        self.dropout_rate = dropout_rate

        self.model_path_root = model_path_root

        self._current_fold = 0
        self._current_bag = 0

    def load_model(self):
        self._model: NnModelWrapper = self.create_model(
            self.model_path_root.joinpath("cv{}/{}/model".format(self._current_fold, self._current_bag)))

    def predict(self, dataset: TorchSimpleDataset, batch_size=256, suffix=""):

        self.suffix = suffix
        self._measure_ids = dataset.measurement_ids

        folds = [int(path.name.replace("cv", "")) for path in self.model_path_root.glob("cv*")]
        self._fold_results = []

        for fold in folds:
            self._current_fold = fold
            self._predict_fold(dataset, batch_size)

        df = pd.DataFrame()
        df["id_measurement"] = self._measure_ids

        df["hard_hard_voted_class"] = self._hard_vote(self._fold_results, class_col="hard_voted_class")
        df["avg_avg_probability"], df["soft_soft_voted_class"] = self._soft_vote(self._fold_results,
                                                                                 probability_col="avg_probability")
        df["hard_soft_voted_class"] = self._hard_vote(self._fold_results, class_col="soft_voted_class")
        df.to_csv(self.model_path_root.joinpath(self.suffix + "predicted.csv"))
        return df

    def _predict_fold(self, dataset: TorchSimpleDataset, batch_size=256):
        self._current_fold_root = self.model_path_root.joinpath("cv{}".format(self._current_fold))

        bags = [int(path.name) for path in self._current_fold_root.glob("*") if self.BAG_ROOT_PATTERN.match(path.name)]
        self._bag_results = []

        for bag in tqdm(bags):
            self._current_bag = bag
            self._current_bag_root = self._current_fold_root.joinpath(str(self._current_bag))
            logger.info("**** predicting with fold {} bag {} ****".format(self._current_fold, self._current_bag))
            self.load_model()
            predicted = self._model.predict(dataset, batch_size, n_cpus)
            df = pd.DataFrame()
            df["id_measurement"] = self._measure_ids
            df["probability"] = predicted
            df["class"] = binarize(predicted, threshold=self.threshold)
            df.to_csv(self._current_bag_root.joinpath(self.suffix + "predicted.csv"))
            self._bag_results.append(df)

        logger.info("**** soft voting with fold {} bag {} ****".format(self._current_fold, self._current_bag))
        self._fold_df = pd.DataFrame()
        self._fold_df["id_measurement"] = self._measure_ids
        self._fold_df["hard_voted_class"] = self._hard_vote(self._bag_results)
        self._fold_df["avg_probability"], self._fold_df["soft_voted_class"] = self._soft_vote(self._bag_results)
        self._fold_df.to_csv(self._current_fold_root.joinpath(self.suffix + "predicted.csv"))
        self._fold_results.append(self._fold_df)

    @abstractmethod
    def create_model(self, model_path):
        pass

    @staticmethod
    def to_submission(df):
        logger.info("creating submission form")
        test_df = VbsDataSetFactory().test_df
        soft_soft_df = df[["id_measurement", "soft_soft_voted_class"]]
        hard_hard_df = df[["id_measurement", "hard_hard_voted_class"]]
        hard_soft_df = df[["id_measurement", "hard_soft_voted_class"]]

        def f(df: pd.DataFrame):
            submission_df = test_df.merge(df, on="id_measurement", how="left")
            submission_df.drop(["id_measurement", "phase"], axis=1, inplace=True)
            col = [str(col) for col in submission_df.columns if str(col) != "signal_id"][0]
            submission_df.rename(columns={col: "target"}, inplace=True)
            submission_df["target"] = submission_df["target"].astype(bool)
            return submission_df

        submission_dfs = {}
        submission_dfs["soft_soft"] = f(soft_soft_df)
        submission_dfs["hard_hard"] = f(hard_hard_df)
        submission_dfs["hard_soft"] = f(hard_soft_df)

        return submission_dfs


class GruTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = Gru(self.model_params,
                            save_dir=save_dir,
                            lr=self.lr, weight_decay=self.weight_decay,
                            optimizer_factory=BiLstmAttension.create_adabound)
        return model_wrapper


class BiGruAttensionTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, loss=None):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, loss=loss)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=save_dir,
                                       lr=self.lr, weight_decay=self.weight_decay, loss_function=self.loss)
        return model_wrapper


class BiGruAttensionPathTrainRunner(PathKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, aug_ratio=0, oversampling=False,
                 loss=None):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, aug_ratio=aug_ratio, oversampling=oversampling,
                         loss=loss)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=save_dir,
                                       lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class BiGruAttensionMixupTrainRunner(MixupKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, mixup_ratio=3):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, mixup_ratio)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=save_dir,
                                       lr=self.lr, weight_decay=self.weight_decay,
                                       optimizer_factory=BiLstmAttension.create_adabound)
        return model_wrapper


class BiLstmAttensionTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = BiLstmAttension(self.model_params,
                                        save_dir=save_dir,
                                        lr=self.lr, weight_decay=self.weight_decay,
                                        optimizer_factory=BiLstmAttension.create_adabound)
        return model_wrapper


class BiGruAttensionBaggingTrainRunner(UndersampleBaggingKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, n_bag=5):
        self.model_params = model_params
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, n_bag)

    def create_model(self, save_dir):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=save_dir,
                                       lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class GruKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = Gru(self.model_params,
                            save_dir=self._current_fold_root,
                            lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper


class BiGruAttensionKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=self._current_fold_root,
                                       lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper


class BiGruAttentionBaggingModel(BaggingKfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = BiGruAttension(self.model_params,
                                       save_dir=self._current_fold_root,
                                       lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class BiLstmAttentionKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = BiLstmAttension(self.model_params,
                                        save_dir=self._current_fold_root,
                                        lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class HerarchicalRawAtteintionTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold)
        self.model_params = model_params

    # violate lsp
    def __call__(self, save_dir, num_workers=None, is_train=True):
        raw_data_set = VbsDataSetFactory()(is_train)
        dummy_feature = raw_data_set.measurement_ids
        super().__call__(raw_data_set, dummy_feature, save_dir, num_workers)

    def create_model(self, save_dir):
        model_wrapper = HierarchicalAttention(self.model_params,
                                              save_dir=save_dir,
                                              lr=self.lr, weight_decay=self.weight_decay, parallel=False)
        return model_wrapper

    def create_dataset(self, train_X, train_Y, train_measurement_ids, valid_X, valid_Y, valid_measurement_ids):
        train_dataset = TorchDelayedRawDataset(self.raw_dataset, train_measurement_ids)
        valid_dataset = TorchDelayedRawDataset(self.raw_dataset, valid_measurement_ids)
        return train_dataset, valid_dataset


class HierarchicalAttentionTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, loss=None):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, loss=loss)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = HierarchicalAttention(self.model_params,
                                              save_dir=save_dir,
                                              lr=self.lr, weight_decay=self.weight_decay, loss_function=self.loss)
        return model_wrapper


class HierarchicalAttentionKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = HierarchicalAttention(self.model_params,
                                              save_dir=self._current_fold_root,
                                              lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper


class SENetTrainRunner(KfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = SENetWrapper(self.model_params,
                                     save_dir=save_dir,
                                     lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class SENetPathTrainRunner(PathKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10,
                 lr=1e-4, weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, aug_ratio=4,
                 oversampling=False):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, aug_ratio, oversampling)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = SENetWrapper(self.model_params,
                                     save_dir=save_dir,
                                     lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class SENetKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = SENetWrapper(self.model_params,
                                     save_dir=self._current_fold_root,
                                     lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper


class CnnAttentionMultipleDropoutTrainRunner(PathKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10,
                 lr=1e-4, weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, aug_ratio=4,
                 oversampling=False):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, aug_ratio, oversampling)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = CnnAttentionMultipleDropoutWrapper(self.model_params,
                                                           save_dir=save_dir,
                                                           lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class BasicCnnAttentionTrainRunner(PathKfoldTrainRunner):

    def __init__(self, model_params, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40,
                 patience=10,
                 lr=1e-4, weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, aug_ratio=4,
                 oversampling=False):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size, n_fold, aug_ratio, oversampling)
        self.model_params = model_params

    def create_model(self, save_dir):
        model_wrapper = BasicCnnAttentionWrapper(self.model_params,
                                                 save_dir=save_dir,
                                                 lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class BasicCnnAttentionKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = BasicCnnAttentionWrapper(self.model_params,
                                                 save_dir=self._current_fold_root,
                                                 lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper


class CnnAttentionMultipleDropoutKfoldModel(KfoldModel):

    def __init__(self, model_params, model_path_root: Path, threshold=0.5):
        super().__init__(model_path_root, threshold)
        self.model_params = model_params

    def create_model(self, model_path):
        model_wrapper = CnnAttentionMultipleDropoutWrapper(self.model_params,
                                                           save_dir=self._current_fold_root,
                                                           lr=0, weight_decay=0, model_path=model_path)

        return model_wrapper
