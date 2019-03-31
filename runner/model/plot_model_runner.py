import logging
import math
import random
import re
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
from imblearn import under_sampling
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import binarize
from tqdm import tqdm

from feature.base import RANDOM_STATE
from feature.plot_images import PlotImageDataset
from model.nn_model_wrapper import ImagenetTransformers, NnModelWrapper, VGG16Wrapper, ResNet50Wrapper, \
    ResNet50ThinFcWrapper, ResNet18PretrainedWrapper, ResNet10Wrapper, XceptionWrapper, SqueezenetWrapper, \
    SeNet154Wrapper
from runner.model.base_runner import logger, EnsembleModel
from utils.data.dataset import n_cpus, VbsDataSetFactory


class PlotTrainRunner(object, metaclass=ABCMeta):
    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10,
                 lr=1e-4, weight_decay=1e-5, is_debug=False, valid_size=0.2):
        self.patience = patience
        self.n_epochs = n_epochs
        self.valid_batch_size = valid_batch_size
        self.train_batch_size = train_batch_size
        self.dropout_rate = dropout_rate
        self.is_debug = is_debug
        self.lr = lr
        self.weight_decay = weight_decay
        self.valid_size = valid_size

    def __call__(self, plot_root_path, save_dir):
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")

        train_plot_meta_df = PlotImageDataset.read_meta_df(plot_root_path)

        train_plot_meta_df, valid_plot_meta_df = self.train_valid_split_df(train_plot_meta_df, self.valid_size)
        transformers = ImagenetTransformers()
        train_dataset = PlotImageDataset(train_plot_meta_df, plot_root_path, transformers=transformers)
        valid_dataset = PlotImageDataset(valid_plot_meta_df, plot_root_path, transformers=transformers)
        # save_dir = Path("/mnt/gcs/kaggle-grasp-and-lift-eeg-detection/model/vgg_pretrained/window_750_stride_75")
        save_dir.mkdir(exist_ok=True, parents=True)

        file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        model_wrapper = self.create_model(save_dir)

        if self.is_debug:
            DEBUG_SIZE = 4000
            train_dataset.train_dataset.label_df = train_dataset.train_dataset.label_df[:DEBUG_SIZE]
            train_dataset.train_dataset.plot_paths = train_dataset.train_dataset.plot_paths[:DEBUG_SIZE]
            train_dataset.valid_dataset.label_df = train_dataset.valid_dataset.label_df[:DEBUG_SIZE]
            train_dataset.valid_dataset.plot_paths = train_dataset.valid_dataset.plot_paths[:DEBUG_SIZE]
        model_wrapper.train(train_dataset, valid_dataset,
                            train_batch_size=self.train_batch_size, valid_batch_size=self.valid_batch_size,
                            n_epochs=self.n_epochs, patience=self.patience,
                            num_workers=n_cpus)

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


class PlotKfoldTrainRunner(PlotTrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size)
        self.n_fold = n_fold

    def __call__(self, plot_root_path, save_dir: Path, transformers=None):
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
        self._plot_root_path = plot_root_path
        self._plot_meta_df = PlotImageDataset.read_meta_df(plot_root_path)

        if not transformers:
            self.transformers = ImagenetTransformers()
        else:
            self.transformers = transformers

        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir

        file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=RANDOM_STATE).split(
            list(range(self._plot_meta_df.shape[0])), self._plot_meta_df["target"],
            groups=self._plot_meta_df["id_measurement"])

        for i, (train_index, test_index) in enumerate(folds):
            logger.info("****cv {} / {} ****".format(i, self.n_fold))
            self._train_cv(i, train_index, test_index)

    def _train_cv(self, i, train_index, test_index):
        self._train_plot_meta_df = self._plot_meta_df.iloc[train_index, :]
        self._valid_plot_meta_df = self._plot_meta_df.iloc[test_index, :]
        cv_root = self.save_dir.joinpath("cv{}".format(i))
        cv_root.mkdir(exist_ok=True, parents=True)
        self._train_plot_meta_df.to_csv(cv_root.joinpath("train.csv"))
        self._valid_plot_meta_df.to_csv(cv_root.joinpath("valid.csv"))
        train_dataset = PlotImageDataset(self._train_plot_meta_df, self._plot_root_path, transformers=self.transformers)
        valid_dataset = PlotImageDataset(self._valid_plot_meta_df, self._plot_root_path,
                                         transformers=self.transformers)

        model_wrapper = self.create_model(cv_root)

        model_wrapper.train(train_dataset, valid_dataset,
                            train_batch_size=self.train_batch_size, valid_batch_size=self.valid_batch_size,
                            n_epochs=self.n_epochs, patience=self.patience,
                            num_workers=n_cpus)

    @abstractmethod
    def create_model(self, save_dir):
        pass


class PlotUndersampleBaggingKfoldTrainRunner(PlotTrainRunner):

    def __init__(self, dropout_rate=0.5, train_batch_size=128, valid_batch_size=128, n_epochs=40, patience=10, lr=1e-4,
                 weight_decay=1e-5, is_debug=False, valid_size=0.1, n_fold=5, n_bag=5):
        super().__init__(dropout_rate, train_batch_size, valid_batch_size, n_epochs, patience, lr, weight_decay,
                         is_debug, valid_size)
        self.n_bag = n_bag
        self.n_fold = n_fold

    def __call__(self, plot_root_path, save_dir: Path, transformers=None):
        # plot_root_path = Path(__file__).parent.parent.parent.parent.joinpath("output/features/train/window_750_stride_75")
        self._plot_root_path = plot_root_path
        self._plot_meta_df = PlotImageDataset.read_meta_df(plot_root_path)

        if not transformers:
            self.transformers = ImagenetTransformers()
        else:
            self.transformers = transformers

        save_dir.mkdir(exist_ok=True, parents=True)
        self.save_dir = save_dir

        file_handler = logging.FileHandler(str(save_dir.joinpath("train.log")))
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        folds = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=RANDOM_STATE).split(
            list(range(self._plot_meta_df.shape[0])), self._plot_meta_df["target"])

        for i, (train_index, test_index) in enumerate(folds):
            logger.info("****cv {} / {} ****".format(i, self.n_fold))
            self._train_cv(i, train_index, test_index)

    def _train_cv(self, i, train_index, test_index):
        self._train_plot_meta_df = self._plot_meta_df.iloc[train_index, :]
        self._valid_plot_meta_df = self._plot_meta_df.iloc[test_index, :]
        cv_root = self.save_dir.joinpath("cv{}".format(i))
        cv_root.mkdir(exist_ok=True, parents=True)
        self._train_plot_meta_df.to_csv(cv_root.joinpath("train.csv"))
        self._valid_plot_meta_df.to_csv(cv_root.joinpath("valid.csv"))
        for j in range(self.n_bag):
            logger.info("---- training bag {} / {} ---- ".format(str(j), self.n_bag))
            bag_root = cv_root.joinpath("{}".format(j))
            bag_root.mkdir(exist_ok=True, parents=True)
            self._train_bag(j, bag_root)

    def _train_bag(self, random_state, bag_root: Path):
        df = self._get_bag(random_state)
        df.to_csv(bag_root.joinpath("bag.csv"))
        train_dataset = PlotImageDataset(df, self._plot_root_path, transformers=self.transformers)
        valid_dataset = PlotImageDataset(self._valid_plot_meta_df, self._plot_root_path, transformers=self.transformers)

        model_wrapper = self.create_model(bag_root)

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
        indices, _ = sampler.fit_sample(self._train_plot_meta_df.index.values.reshape((-1, 1)),
                                        self._train_plot_meta_df["target"])
        return self._train_plot_meta_df.loc[indices.reshape((-1)).tolist()]

    @abstractmethod
    def create_model(self, save_dir):
        pass


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

    def predict(self, plot_root: Path, batch_size=256):
        dataset = PlotImageDataset(PlotImageDataset.read_meta_df(plot_root), plot_root, ImagenetTransformers())
        self._measure_ids = dataset.plot_meta_df["id_measurement"]

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
        df.to_csv(self.model_path_root.joinpath("predicted.csv"))
        return df

    def _predict_fold(self, dataset: PlotImageDataset, batch_size=256):
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
            df.to_csv(self._current_bag_root.joinpath("predicted.csv"))
            self._bag_results.append(df)

        logger.info("**** soft voting with fold {} bag {} ****".format(self._current_fold, self._current_bag))
        self._fold_df = pd.DataFrame()
        self._fold_df["id_measurement"] = self._measure_ids
        self._fold_df["hard_voted_class"] = self._hard_vote(self._bag_results)
        self._fold_df["avg_probability"], self._fold_df["soft_voted_class"] = self._soft_vote(self._bag_results)
        self._fold_df.to_csv(self._current_fold_root.joinpath("predicted.csv"))
        self._fold_results.append(self._fold_df)

    @abstractmethod
    def create_model(self, model_path):
        pass

    @staticmethod
    def to_submission(df):
        logger.info("creating submission form")
        test_df = VbsDataSetFactory().test_df
        soft_soft_df = df[["id_measurement", "soff_soft_voted_class"]]
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


class PlotKfoldModel(EnsembleModel, metaclass=ABCMeta):

    def __init__(self, model_path_root: Path, dropout_rate=0.5, threshold=0.5):
        super().__init__(threshold)
        self.dropout_rate = dropout_rate

        self.model_path_root = model_path_root

        self._current_fold = 0

    def load_model(self):
        self._model: NnModelWrapper = self.create_model(
            self.model_path_root.joinpath("cv{}/model".format(self._current_fold)))

    def predict(self, plot_root: Path, batch_size=256, transformers=None):
        if not transformers:
            transformers = ImagenetTransformers()
        dataset = PlotImageDataset(PlotImageDataset.read_meta_df(plot_root), plot_root, transformers)
        self._measure_ids = dataset.plot_meta_df["id_measurement"]

        folds = [int(path.name.replace("cv", "")) for path in self.model_path_root.glob("cv*")]
        self._fold_results = []

        for fold in folds:
            self._current_fold = fold
            self._predict_fold(dataset, batch_size)

        self._fold_df = pd.DataFrame()
        self._fold_df["id_measurement"] = self._measure_ids
        self._fold_df["hard_voted_class"] = self._hard_vote(self._fold_results)
        self._fold_df["avg_probability"], self._fold_df["soft_voted_class"] = self._soft_vote(self._fold_results)
        self._fold_df.to_csv(self.model_path_root.joinpath("predicted.csv"))
        self._fold_results.append(self._fold_df)

        self._fold_df.to_csv(self.model_path_root.joinpath("predicted.csv"))
        return self._fold_df

    def _predict_fold(self, dataset: PlotImageDataset, batch_size=256):
        self._current_fold_root = self.model_path_root.joinpath("cv{}".format(self._current_fold))

        logger.info("**** predicting with fold {} ****".format(self._current_fold))
        self.load_model()
        predicted = self._model.predict(dataset, batch_size, n_cpus)
        df = pd.DataFrame()
        df["id_measurement"] = self._measure_ids
        df["probability"] = predicted
        df["class"] = binarize(predicted, threshold=self.threshold)
        df.to_csv(self._current_fold_root.joinpath("predicted.csv"))
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


class Vgg16PretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = VGG16Wrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                     lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet50PretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet50Wrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                        lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet50ThinFcPretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet50ThinFcWrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                              lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet18PretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet18PretrainedWrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                                  lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet18UndersampleBaggingKfoldTrainRunner(PlotUndersampleBaggingKfoldTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet18PretrainedWrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                                  lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet18KfoldTrainRunner(PlotKfoldTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet18PretrainedWrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                                  lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet10KfoldTrainRunner(PlotKfoldTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = ResNet10Wrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                        lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class ResNet18BaggingKfoldModel(BaggingKfoldModel):

    def create_model(self, model_path):
        model_wrapper = ResNet18PretrainedWrapper(1, dropout_rate=self.dropout_rate, save_dir=self._current_bag_root,
                                                  lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class ResNet18PlotKfoldModel(PlotKfoldModel):

    def create_model(self, model_path):
        model_wrapper = ResNet18PretrainedWrapper(1, dropout_rate=self.dropout_rate, save_dir=self._current_fold_root,
                                                  lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class ResNet10PlotKfoldModel(PlotKfoldModel):

    def create_model(self, model_path):
        model_wrapper = ResNet10Wrapper(1, dropout_rate=self.dropout_rate, save_dir=self._current_fold_root,
                                        lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class XceptionUndersampleBaggingKfoldTrainRunner(PlotUndersampleBaggingKfoldTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = XceptionWrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                        lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper


class XceptionBaggingKfoldModel(BaggingKfoldModel):

    def create_model(self, model_path):
        model_wrapper = XceptionWrapper(1, dropout_rate=self.dropout_rate, save_dir=self._current_bag_root,
                                        lr=0, weight_decay=0, model_path=model_path)
        return model_wrapper


class SqueezenetPretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = SqueezenetWrapper(1, save_dir=save_dir, lr=self.lr, weight_decay=self.weight_decay)

        return model_wrapper


class SeNet154PretrainedTrainRunner(PlotTrainRunner):

    def create_model(self, save_dir):
        model_wrapper = SeNet154Wrapper(1, dropout_rate=self.dropout_rate, save_dir=save_dir,
                                        lr=self.lr, weight_decay=self.weight_decay)
        return model_wrapper
