import os
from abc import ABCMeta, abstractmethod

import matplotlib as mpl
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from feature.base import Feature
from model.nn_model_wrapper import RANDOM_SEED, ImagenetTransformers

mpl.use('Agg')
import multiprocessing as mp
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import skimage.io
import torch.utils.data
from utils.data.dataset import VbsMeasurement, VbsDataSet
import pretrainedmodels

n_cpus = os.cpu_count()


class PlotWriter(object):
    FILE_PATH_PATTERN = "measurement_{}_window_{}_start_{}_end_{}.png"
    PLOT_META_FILE = "plot_meta.csv"

    def __init__(self, save_root_dir: Path):
        self.root_dir = save_root_dir
        self._save_dir = None
        self._figsize = None
        self._current_save_dir = None
        self.n_jobs = -1

    def write(self, dataset: VbsDataSet, window_size, stride_size, figsize, n_jobs=None):
        # self._dataset = dataset
        self._datagenerator = dataset
        self._save_dir = self.root_dir.joinpath("window_{}_stride_{}".format(window_size, stride_size))
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._window_size = window_size
        self._stride_size = stride_size
        self._figsize = figsize
        self.n_jobs = n_jobs
        self._current_df = None
        self._subject = None
        self._series = None
        columns = ["id_measurement", "file_path", "window_idx", "start", "end"]

        if dataset.is_train:
            columns.append("target")
        self._plot_meta_df = pd.DataFrame(columns=columns)
        self._is_train = dataset.is_train

        for measurement in dataset.measurements:
            print("writing plot of windows from measurement {}".format(measurement.measurement_id))
            self._write_measurement(measurement)

        self._plot_meta_df.to_csv(self._save_dir.joinpath(self.PLOT_META_FILE), index=None)

    def _write_measurement(self, measurement: VbsMeasurement):
        self._current_save_dir = self._save_dir.joinpath(str(measurement.measurement_id))
        self._current_save_dir.mkdir(parents=True, exist_ok=True)

        self._current_df = measurement.load_as_df()
        self._current_measurement_id = measurement.measurement_id

        indices = [(i, self._stride_size * i, self._stride_size * i + self._window_size)
                   for i in range((self._current_df.shape[0] - self._window_size + 1) // self._stride_size)]
        if not (self._current_df.shape[0] - self._stride_size * len(indices)) % self._window_size:
            indices += [(len(indices), self._stride_size * len(indices), self._current_df.shape[0])]

        with mp.get_context("spawn").Pool(self.n_jobs) as pool:
            # tqdm(pool.imap_unordered(self._write_window, indices))
            windows = pool.map(self._write_window, indices)

        rows = [self._to_meta_row(measurement, window) for window in windows]
        self._plot_meta_df = self._plot_meta_df.append(rows, ignore_index=True)

    #
    # @staticmethod
    # def _pad_df_with_first_value(df, pad_size):
    #     return pd.concat([df.iloc[0:1] for _ in range(pad_size)] + [df], axis=0, ignore_index=True)

    def _to_meta_row(self, measurement, window):
        row = {"id_measurement": self._current_measurement_id, "file_path": window[0],
               "window_idx": window[1], "start": window[2], "end": window[3]}
        if self._is_train:
            row["target"] = measurement.target
        return row

    def _write_window(self, window_start_end_idx):
        window_idx, start, end = window_start_end_idx

        fig = plt.figure(random.randint(0, 100))
        dpi = fig.get_dpi()
        fig.set_size_inches(self._figsize[0] / dpi, self._figsize[1] / dpi)

        axes = self._current_df.iloc[start:end].plot(
            subplots=True, sharex=True, sharey=True,
            figsize=(self._figsize[0] / dpi, self._figsize[1] / dpi)
        )
        for ax in axes:
            ax.axis("off")
            ax.legend().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        file_name = self.FILE_PATH_PATTERN.format(self._current_measurement_id, window_idx, start, end)
        plt.savefig(self._current_save_dir.joinpath(file_name),
                    bbox_inches='tight', dpi=dpi, transparent=True, pad_inches=0.0)
        plt.close("all")
        return file_name, window_idx, start, end


class PlotImageDataset(torch.utils.data.Dataset):

    def __init__(self, plot_meta_df: pd.DataFrame, plot_root: Path, transformers):
        self.plot_meta_df = plot_meta_df
        self.plot_root = plot_root
        self.transformers = transformers
        self.is_train = "target" in self.plot_meta_df
        if self.is_train:
            self.plot_meta_df["target"] = self.plot_meta_df["target"].astype("float32")

    def __getitem__(self, index):
        image = skimage.io.imread(str(self._get_file_path(self.plot_meta_df.iloc[index])))[:, :, :3]
        if self.transformers:
            image = self.transformers(image)
        item = {"image": image}
        if not self.is_train:
            return item
        item["label"] = self.plot_meta_df["target"][index:index + 1].values
        return item

    def _get_file_path(self, row):
        return self.plot_root.joinpath("{}/{}".format(row["id_measurement"], row["file_path"]))

    def __len__(self):
        return self.plot_meta_df.shape[0]

    @staticmethod
    def read_meta_df(plot_root: Path):
        return pd.read_csv(plot_root.joinpath(PlotWriter.PLOT_META_FILE))


class FrozenDnnFeature(Feature, metaclass=ABCMeta):

    def __init__(self):
        self.model_name = self.get_model_name()

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_model_name(self):
        pass

    def transform(self, dataset: PlotImageDataset, save_path=None, batch_size=256, n_workers=None):
        self._dataset = dataset
        if n_workers is None:
            n_workers = n_cpus
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        outputs = []
        for i, x in tqdm(enumerate(dataloader)):
            x = self.predict(x["image"])
            outputs.append(x.cpu().detach().numpy().reshape((x.shape[0], -1)))
        outputs = np.vstack(outputs)

        feature_columns = [self.model_name + "_" + str(i) for i in range(outputs.shape[1])]
        df = pd.DataFrame(columns=["id_measurement"] + feature_columns)
        df["id_measurement"] = dataset.plot_meta_df.id_measurement
        df[feature_columns] = outputs
        if save_path:
            print("writing feature....")
            df.to_csv(save_path.joinpath("feature.csv"), index=None)
        return df

    def from_plots(self, plot_root, save_path=None, batch_size=256, n_workers=None):
        dataset = PlotImageDataset(PlotImageDataset.read_meta_df(plot_root), plot_root, ImagenetTransformers())
        return self.transform(dataset, batch_size=batch_size, save_path=save_path, n_workers=n_workers)


class FrozenResnet18Feature(FrozenDnnFeature):

    def predict(self, x):
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet = nn.Sequential(*list(resnet.children())[:-1])
        for param in resnet.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            resnet = torch.nn.DataParallel(resnet).cuda()

        resnet.eval()
        # for module in list(resnet.modules())[:-1]:
        #     x = module.forward(x)
        # print(resnet(x))
        # raise ValueError()
        x = resnet(x)
        return x.view(x.size(0), -1)

    def get_model_name(self):
        return "resnet18"


class FrozenXceptionFeature(FrozenDnnFeature):

    def predict(self, x):
        xception = pretrainedmodels.models.xception()
        xception = nn.Sequential(*list(xception.children())[:-1])
        for param in xception.parameters():
            param.requires_grad = False

        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
            xception = torch.nn.DataParallel(xception).cuda()

        xception.eval()
        # for module in list(resnet.modules())[:-1]:
        #     x = module.forward(x)
        # print(resnet(x))
        # raise ValueError()
        x = xception(x)
        return x

    def get_model_name(self):
        return "xception"
