from abc import ABCMeta
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Union

import numpy as np  # linear algebra
import pandas as pd
import pyarrow.parquet as pq
import pywt
from scipy import signal
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from utils.data.dataset import VbsDataSet, n_cpus

RANDOM_STATE = 10


class Feature(object, metaclass=ABCMeta):

    def __init__(self, transformer, phase_specific=False):
        self.transformer = transformer
        self._current_part = 0
        self.switch_indices = None
        self.phase_specific = phase_specific

    def _transform_signal(self, parquet_path, signal_id):
        signal = read_column(parquet_path, signal_id).values

        return self.transformer.transform(signal)

    def _transform_measurement(self, measure_id):
        if not self.phase_specific:
            return np.hstack(
                [self._transform_signal(self._parquet_path, signal_id) for phase_id, signal_id
                 in enumerate(self._meta_df[self._meta_df["id_measurement"] == measure_id].signal_id)
                 ]
            )

        temp = np.vstack([read_column(self._parquet_path, signal_id).values for phase_id, signal_id
                          in enumerate(self._meta_df[self._meta_df["id_measurement"] == measure_id].signal_id)])
        temp = temp.reshape([-1] + list(temp.shape))
        temp = self.transformer.transform(temp)
        return np.squeeze(temp, axis=0)

    def fit_transform(self, dataset: Union[VbsDataSet, 'Feature', Path], save_dir=None, n_jobs=None):
        return self.fit(dataset, save_dir, n_jobs).transform(dataset, save_dir, n_jobs)

    def fit(self, dataset: Union[VbsDataSet, 'Feature', Path], save_dir=None, n_jobs=None):
        if isinstance(dataset, VbsDataSet):
            x = dataset.get_stacked_matrix()

        else:
            raise ValueError("not implemented")

        self.transformer.fit(x)
        # x = x.reshpe(src_shape)
        joblib.dump(self.transformer, str(save_dir.joinpath("transformer.pickle")))
        return self

    def transform(self, dataset: Union[VbsDataSet, 'Feature'], save_dir=None, n_jobs=-1):
        self._parquet_path = dataset.parquet_path
        self._meta_df = dataset.meta_df
        if n_jobs is None or n_jobs < 0:
            n_jobs = n_cpus

        self.split_border = 3000
        if len(dataset.measurement_ids) < self.split_border:
            with Pool(n_jobs) as pool:
                rows = pool.map(self._transform_measurement, dataset.measurement_ids)
            rows = np.stack(rows)
            joblib.dump(rows, str(save_dir.joinpath("feature.pickle")))
        else:
            split_size = self.split_border // 2
            n_split = len(dataset.measurement_ids) // split_size
            n_split += int(len(dataset.measurement_ids) % split_size > 0)
            for i in range(n_split):
                start = i * split_size
                end = (i + 1) * split_size
                with Pool(n_jobs) as pool:
                    if i < n_split - 1:
                        splited_measurement_ids = dataset.measurement_ids[start:end]
                    else:
                        splited_measurement_ids = dataset.measurement_ids[start:]

                    rows = pool.map(self._transform_measurement, splited_measurement_ids)
                    rows = np.stack(rows)
                    joblib.dump(rows, str(save_dir.joinpath("feature_{}.pickle".format(i))))

    def load(self, save_path, switch_indices=None):
        if isinstance(save_path, list):
            self.save_paths = list(sorted(save_path))
            self.switch_indices = switch_indices
            # TODO How to load?
        elif save_path.suffix == ".csv":
            self.feature_df = pd.read_csv(save_path)
            self.feature_df.sort_values(by="id_measurement", inplace=True, axis=0)
            self.feature_df.reset_index(inplace=True)
        elif save_path.suffix == ".pickle":
            self.feature_array = joblib.load(save_path)
            if switch_indices is not None:
                self.feature_array = self.feature_array[:, :, switch_indices]
        return self

    def partial_load(self, n=None):
        if n is not None:
            self.feature_array = joblib.load(self.save_paths[n])
        else:
            while self._current_part < len(self.save_paths):
                x = joblib.load(self.save_paths[self._current_part])
                if self.switch_indices is not None:
                    x = x[:, :, self.switch_indices]
                yield x
                self._current_part += 1

    def get_all_rows(self):
        return np.vstack([joblib.load(path) for path in self.save_paths])

    def get_all_raw_rows(self):
        return np.vstack([joblib.load(path) for path in self.save_paths if "raw" in path.stem])

    def flash(self):
        self._current_part = 0


class WindowFeature(Feature):

    def to_array(self):
        return self.feature_array

    def __init__(self, transformer, window_size, step_size, accept_window=False, output_dim=None, pad=False, axis=0):
        super().__init__(transformer)
        self.transformer = transformer
        self._parquet_path = None
        self._meta_df = None
        self.window_size = window_size
        self.step_size = step_size
        self.accept_window = accept_window
        self.ouput_dim = output_dim
        self.pad = pad
        self.axis = axis

    def fit_transform(self, dataset: Union[VbsDataSet, 'Feature'], save_dir=None, n_jobs=None):
        return self.fit(dataset, save_dir, n_jobs).transform(dataset, save_dir, n_jobs)

    def fit(self, dataset: Union[VbsDataSet, 'Feature', Path], save_dir=None, n_jobs=None):
        if isinstance(dataset, VbsDataSet):
            x = dataset.get_stacked_matrix()

        if isinstance(dataset, Path):
            x = self._fit_from_files(dataset, n_jobs)
        elif self.accept_window:
            x = dataset.feature_array
            src_shape = x.shape
            x = x.reshape((-1, src_shape[-1]))
        else:
            raise ValueError("not implemented")

        self.transformer.fit(x)
        # x = x.reshpe(src_shape)
        joblib.dump(self.transformer, str(save_dir.joinpath("transformer.pickle")))
        return self

    def _fit_from_files(self, path, n_jobs=None):
        raise ValueError("not implemented")

    #
    # def _transform_window(values):
    #     return transformer.transform(read_column(parquet_path, signal_id).values)

    def _transform_signal(self, parquet_path, signal_id):
        signal = read_column(parquet_path, signal_id).values
        return self._to_transformed_windows(signal)

    def _to_transformed_windows(self, signal):
        n_windows = (signal.shape[self.axis] - self.window_size + 1) // self.step_size
        n_windows += int(not bool((signal.shape[self.axis] - self.step_size * n_windows)
                                  % (self.window_size)))
        # element_bit = signal.dtype.itemsize * 8
        #         window_views = np.lib.stride_tricks.as_strided(signal, (n_windows, self.window_size),
        #                                                      (self.step_size * element_bit, element_bit))
        window_indices = [(i * self.step_size, i * self.step_size + self.window_size)
                          for i in range(n_windows)]
        return np.asarray(
            [self.transformer.transform(self._get_window(signal, start, end)) for start, end in window_indices])

    def _get_window(self, x, start, end):
        if self.axis == 0:
            return x[start:end]
        else:
            return x[:, start:end]

    def _transform_measurement(self, measure_id):
        temp = np.hstack(
            [self._transform_signal(self._parquet_path, signal_id) for signal_id
             in self._meta_df[self._meta_df["id_measurement"] == measure_id].signal_id
             ]
        )
        if self.pad:
            temp = np.pad(temp, pad_width=((1, 0), (0, 0)), mode="constant", constant_values=0)

        return temp

    def transform(self, dataset: Union[VbsDataSet, 'Feature', Path], save_dir=None, n_jobs=None):
        if isinstance(dataset, Path):
            self._transform_from_files(dataset, save_dir, n_jobs)
            return
        elif isinstance(dataset, VbsDataSet):
            return self._transform_dataset(dataset, n_jobs, save_dir)

        elif isinstance(dataset, Feature):
            return self._transform_feature(dataset, save_dir, self.accept_window)

    def _transform_feature(self, dataset: Feature, save_dir, accept_window=False):
        if not hasattr(dataset, "save_paths"):
            x = dataset.feature_array
            if accept_window:
                x = self._transform_matrix(save_dir, x)
            else:
                x = self._to_transformed_windows(x)
                joblib.dump(x, str(save_dir.joinpath("feature.pickle")))

            return x

        for i, _ in enumerate(dataset.save_paths):
            dataset.partial_load(i)
            if accept_window:
                self._transform_matrix(save_dir, dataset.feature_array, str(i))
            else:
                x = self._to_transformed_windows(dataset.feature_array)
                joblib.dump(x, str(save_dir.joinpath("feature_" + str(i) + ".pickle")))

    def _transform_matrix(self, save_dir, x, n=""):
        src_shape = x.shape
        x = x.reshape((-1, x.shape[-1]))
        x = self.transformer.transform(x)
        if self.ouput_dim is not None:
            x = x.reshape(list(src_shape[:2]) + [self.ouput_dim])
        else:
            x = x.reshape(src_shape)
        joblib.dump(x, str(save_dir.joinpath("feature_" + n + ".pickle")))
        return x

    def _transform_dataset(self, dataset, n_jobs, save_dir):
        self._parquet_path = dataset.parquet_path
        self._meta_df = dataset.meta_df
        if n_jobs is None:
            n_jobs = n_cpus
        self.split_border = 3000
        if len(dataset.measurement_ids) < self.split_border:
            with Pool(n_jobs) as pool:
                rows = pool.map(self._transform_measurement, dataset.measurement_ids)
            rows = np.stack(rows)
            joblib.dump(rows, str(save_dir.joinpath("feature.pickle")))
        else:
            split_size = self.split_border // 2
            n_split = len(dataset.measurement_ids) // split_size
            n_split += int(len(dataset.measurement_ids) % split_size > 0)
            for i in range(n_split):
                start = i * split_size
                end = (i + 1) * split_size
                with Pool(n_jobs) as pool:
                    if i < n_split - 1:
                        splited_measurement_ids = dataset.measurement_ids[start:end]
                    else:
                        splited_measurement_ids = dataset.measurement_ids[start:]

                    rows = pool.map(self._transform_measurement, splited_measurement_ids)
                    rows = np.stack(rows)
                    joblib.dump(rows, str(save_dir.joinpath("feature_{}.pickle".format(i))))

    def _transform_from_files(self, root_dir, save_dir, n_jobs):
        raw_pickle_paths = list(root_dir.glob("**/*.pickle"))
        self.save_dir = save_dir
        self.raw_dir = root_dir
        with Pool(n_jobs) as pool:
            pool.map(self._transform_file, raw_pickle_paths)

    def _transform_file(self, path):
        save_path = self.save_dir.joinpath(str(path.relative_to(self.raw_dir)))
        save_path.parent.mkdir(exist_ok=True, parents=True)
        x = self._to_transformed_windows(joblib.load(str(path)))
        joblib.dump(x, str(save_path))

    def _transform_(self, root_dir, save_dir, n_jobs):
        raw_pickle_paths = list(root_dir.glob("**/*.pickle"))
        self.save_dir = save_dir
        self.raw_dir = root_dir
        with Pool(n_jobs) as pool:
            pool.map(self._transform_file, raw_pickle_paths)


class WindowScaledFeature(WindowFeature, metaclass=ABCMeta):

    def __init__(self, transformer, window_size, step_size):
        super().__init__(transformer, window_size, step_size)
        self.scaler = None

    def transform(self, dataset: VbsDataSet, save_dir=None, n_jobs=None):
        rows = super().transform(dataset, save_dir, n_jobs)
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(rows)
            joblib.dump(self, str(save_dir.joinpath("feature_object.pickle")))
        rows = self.scaler.transform(rows)
        joblib.dump(rows, str(save_dir.joinpath("feature_scaled.pickle")))
        return rows


class Features(object):

    def __init__(self, feature_classes):
        self.list = [feature() for feature in feature_classes]

    def load(self, save_paths):
        features = [feature.load(path) for feature, path in zip(self.list, save_paths)]
        id_measurement = features[0].feature_df.id_measurement
        features = [feature.feature_df.drop(["id_measurement", "index"], axis=1) for feature in features]
        features = pd.concat([id_measurement] + features, axis=1)
        self.df = features
        return self

    def get_values(self):
        return self.df.drop(["id_measurement"], axis=1).values


class PhaseStandardScaled(Feature):
    N_PHASE = 3

    def __init__(self):
        self.transformers = [StandardScaler() for _ in range(3)]

    def fit_transform(self, dataset: Union[VbsDataSet, 'Feature', Path], save_dir=None, n_jobs=None):
        for i, transformer in enumerate(self.transformers):
            phase = i
            transformer.fit(dataset.get_flat_signals_in_phase(phase))
        pickle_path = save_dir.joinpath("transformer.pickle")
        joblib.dump(self.transformers, str(pickle_path))

        return self.transform(dataset, save_dir)


class WindowStandardScaled(WindowFeature):

    def __init__(self, window_size, step_size, accept_window=False, output_dim=None):
        super().__init__(StandardScaler(), window_size, step_size, accept_window, output_dim)


class PhaseStandardScaler(TransformerMixin):

    def __init__(self, n_phase=3):
        self.n_pahse = n_phase
        self.scalers = [StandardScaler() for _ in range(3)]

    def fit(self, x, y=None):
        assert x.shape[1] == len(self.scalers)
        for i, scaler in enumerate(self.scalers):
            scaler.fit(x[:, i].reshape((-1, 1)))
        return self

    def transform(self, X):
        return np.stack(
            [scaler.transform(X[:, i].reshape((-1, 1))).astype("float32").reshape((X.shape[0], -1)) for i, scaler in
             enumerate(self.scalers)], axis=1).astype("float32")


class WindowPhaseStandardScaler(WindowFeature):

    def __init__(self, window_size, step_size, accept_window=False, output_dim=None):
        super().__init__(PhaseStandardScaler(), window_size, step_size, accept_window, output_dim)


class PhaseStandardScaledFeature(Feature):

    def __init__(self):
        super().__init__(PhaseStandardScaler(), phase_specific=True)


# This function standardize the data from (-128 to 127) to (-1 to 1)
# Theoretically it helps in the NN Model training, but I didn't tested without it
def min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


class PercentileSummaryTransformer(FunctionTransformer):
    max_num = 127
    min_num = -128

    def __init__(self, axis=0, should_flat=True,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.axis = axis
        self.should_flat = should_flat
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        # def transform_ts(ts, n_dim=160, min_max=(-1, 1)):
        # convert data into -1 to 1
        # ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
        # # bucket or chunk size, 5000 in this case (800000 / 160)
        # bucket_size = int(sample_size / n_dim)
        # # new_ts will be the container of the new data
        # new_ts = []
        # this for iteract any chunk/bucket until reach the whole sample_size (800000)
        # for i in range(0, sample_size, bucket_size):
        #     # cut each bucket to ts_range
        #     ts_range = ts_std[i:i + bucket_size]
        #     # calculate each feature
        X = min_max_transf(X, min_data=self.min_num, max_data=self.max_num)
        return self.percentile_summarize(X)

    def percentile_summarize(self, X):
        X = min_max_transf(X, min_data=self.min_num, max_data=self.max_num)
        mean = X.mean(axis=self.axis)
        std = X.std(axis=self.axis)  # standard deviation
        std_top = mean + std  # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentile_calc = np.percentile(X, [0, 1, 25, 50, 75, 99, 100], axis=self.axis)
        max_range = percentile_calc[-1] - percentile_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentile_calc - mean  # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        if not self.axis:
            summary = np.hstack(
                [np.asarray([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile])
        elif self.axis == 1:
            summary = np.vstack(
                [np.vstack([mean, std, std_top, std_bot, max_range]), percentile_calc, relative_percentile]).transpose()
        else:
            ValueError("not implementes")
        if self.should_flat:
            return summary.flatten()
        return summary


class PercentileSummaryWindowFeature(WindowFeature):

    def __init__(self, window_size, step_size, axis=0, should_flat=True):
        super().__init__(PercentileSummaryTransformer(axis=axis,
                                                      should_flat=should_flat), window_size, step_size, axis=1)


class SummaryTransformer(FunctionTransformer):
    max_num = 127
    min_num = -128

    def __init__(self, axis=0, scale="minmax",
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        self.scale = scale
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)
        self.axis = axis

    def f(self, X):
        if self.scale == "minmax":
            X = min_max_transf(X, min_data=self.min_num, max_data=self.max_num)
        avgs = np.mean(X, axis=self.axis)
        stds = np.std(X, axis=self.axis)
        maxs = np.max(X, axis=self.axis)
        mins = np.min(X, axis=self.axis)
        medians = np.median(X, axis=self.axis)
        return np.hstack([avgs, stds, maxs, mins, medians])


class AverageTransformer(FunctionTransformer):
    max_num = 127
    min_num = -128

    def __init__(self,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        X = min_max_transf(X, min_data=self.min_num, max_data=self.max_num)
        avgs = np.mean(X, axis=1)
        return avgs.reshape((-1, 1))


class SummaryWindowFeature(WindowFeature):

    def __init__(self, window_size, step_size, axis=0, scale="minmax"):
        super().__init__(SummaryTransformer(axis, scale=scale), window_size, step_size, axis=axis)


class AverageWindowFeature(WindowFeature):

    def __init__(self, window_size, step_size, axis=0):
        super().__init__(AverageTransformer(axis), window_size, step_size, axis=axis)


class WaevletSummaryTransformer(FunctionTransformer):
    def __init__(self, wavelet_width,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.wavelet_width = wavelet_width
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        #         wavelets = signal.cwt(X, signal.ricker, np.arange(1, self.wavelet_width + 1))
        wavelets, _ = pywt.cwt(X, np.arange(1, self.wavelet_width + 1), 'mexh')
        avgs = np.mean(wavelets, axis=1)
        stds = np.std(wavelets, axis=1)
        maxs = np.max(wavelets, axis=1)
        mins = np.min(wavelets, axis=1)
        medians = np.median(wavelets, axis=1)
        return np.concatenate([avgs, stds, maxs, mins, medians])


class SpectrogramSummaryTransformer(FunctionTransformer):
    N_MEASUREMENTS = 800000

    # In[ ]:

    TOTAL_DURATION = 20e-3

    # In[ ]:

    DEFAULT_SAMPLE_RATE = N_MEASUREMENTS / TOTAL_DURATION

    def __init__(self, fft_length, stride_length,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.fft_length = fft_length
        self.stride_length = stride_length
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        X = self.to_spectrogram(X)
        #         print(X)
        # avgs = np.mean(X, axis=1)
        # stds = np.std(X, axis=1)
        # maxs = np.max(X, axis=1)
        # mins = np.min(X, axis=1)
        # medians = np.median(X, axis=1)

        mean = np.mean(X, axis=1).reshape((-1, 1))
        std = np.std(X, axis=1).reshape((-1, 1))  # standard deviation
        # std_top = mean + std  # I have to test it more, but is is like a band
        # std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentile_calc = np.percentile(X, [0, 1, 25, 50, 75, 99, 100], axis=1).transpose()
        max_range = (percentile_calc[:, -1] - percentile_calc[:, 0]).reshape(
            (-1, 1))  # this is the amplitude of the chunk
        relative_percentile = percentile_calc - mean.reshape((-1, 1))  # maybe it could heap to understand the asymmetry
        # now, we just add all the features to new_ts and convert it to np.array
        return np.hstack([mean, std, percentile_calc, max_range, relative_percentile]).flatten()

    def to_spectrogram(self, series):
        f, t, Sxx = signal.spectrogram(series, fs=self.sample_rate, nperseg=self.fft_length,
                                       noverlap=self.fft_length - self.stride_length, window="hanning", axis=0,
                                       return_onesided=True, mode="magnitude", scaling="density")
        return Sxx


class SpectrogramTransformer(FunctionTransformer):
    N_MEASUREMENTS = 800000

    # In[ ]:

    TOTAL_DURATION = 20e-3

    # In[ ]:

    DEFAULT_SAMPLE_RATE = N_MEASUREMENTS / TOTAL_DURATION

    def __init__(self, fft_length, stride_length,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.sample_rate = self.DEFAULT_SAMPLE_RATE
        self.fft_length = fft_length
        self.stride_length = stride_length
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        return self.to_spectrogram(X)

    def to_spectrogram(self, series):
        f, t, Sxx = signal.spectrogram(series, fs=self.sample_rate, nperseg=self.fft_length,
                                       noverlap=self.fft_length - self.stride_length, window="hanning", axis=0,
                                       return_onesided=True, mode="magnitude", scaling="density")
        return Sxx.transpose()


class WaeveletTransformer(FunctionTransformer):
    def __init__(self, wavelet_width,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.wavelet_width = wavelet_width
        super().__init__(self.f, inverse_func, validate, accept_sparse, pass_y, kw_args, inv_kw_args)

    def f(self, X):
        #         wavelets = signal.cwt(X, signal.ricker, np.arange(1, self.wavelet_width + 1))
        wavelets, _ = pywt.cwt(X, np.arange(1, self.wavelet_width + 1), 'mexh')
        return wavelets


class WaveletFeature(Feature):
    def __init__(self, avelet_width):
        super().__init__(WaeveletTransformer(avelet_width))


class FftSummaryFeature(WindowFeature):

    def __init__(self, window_size, step_size, fft_length, fft_stride):
        super().__init__(SpectrogramSummaryTransformer(fft_length, fft_stride), window_size, step_size)


class FftFeature(Feature):

    def __init__(self, fft_length, fft_stride):
        super().__init__(SpectrogramTransformer(fft_length, fft_stride))

    def transform(self, dataset: Union[VbsDataSet, 'Feature'], save_dir=None, n_jobs=-1):
        super().transform(dataset, save_dir, n_jobs)


class PcaFeature(WindowFeature):

    def __init__(self, n_components):
        dummy_window = 200
        dummy_step = 100
        super().__init__(PCA(n_components), window_size=dummy_window, step_size=dummy_step, accept_window=True,
                         output_dim=n_components)


def read_column(parquet_path, column_id):
    return pq.read_pandas(parquet_path, columns=[str(column_id)]).to_pandas()[str(column_id)]


#
# # In[ ]:
#
#
# class WindowFeatureExtractor(object):
#     def __init__(self, transformers, window_size, step_size):
#         self.transformers: List[TransformerMixin] = transformers
#         self._parquet_path = None
#         self._meta_df = None
#         self.window_size = window_size
#         self.step_size = step_size
#
#     def fit(self, parquet_path, meta_df):
#         pass
#
#     #
#     # def _transform_window(values):
#     #     return transformer.transform(read_column(parquet_path, signal_id).values)
#
#     def _transform_signal(self, parquet_path, signal_id):
#         signal = read_column(parquet_path, signal_id).values
#         n_windows = (signal.shape[0] - self.window_size + 1) // self.step_size
#         n_windows += int(not bool((signal.shape[0] - self.step_size * n_windows)
#                                   % (self.window_size)))
#
#         element_bit = signal.dtype.itemsize * 8
#         #         window_views = np.lib.stride_tricks.as_strided(signal, (n_windows, self.window_size),
#         #                                                      (self.step_size * element_bit, element_bit))
#         window_indices = [(i * self.step_size, i * self.step_size + self.window_size)
#                           for i in range(n_windows)]
#         return np.asarray([[transformer.transform(signal[start:end]) for start, end in window_indices]
#                            for transformer in self.transformers]).flatten()
#
#     def _transform_measurement(self, measure_id):
#         temp = np.concatenate(
#             [self._transform_signal(self._parquet_path, signal_id) for signal_id
#              in self._meta_df[self._meta_df["id_measurement"] == measure_id].signal_id
#              ]
#         )
#         return temp
#
#     def transform(self, parquet_path, meta_df, n_jobs=2):
#         self._parquet_path = parquet_path
#         self._meta_df = meta_df
#         with Pool(n_jobs) as pool:
#             rows = pool.map(self._transform_measurement, self._meta_df.id_measurement.unique())
#         #         rows = list(map(self._transform_measurement, self._meta_df.id_measurement.unique()))
#         return np.vstack(rows)


# In[ ]:


# wavelet transform takes too much time
# extractor = FeatureExtractor([SummaryTransformer(), WaevletSummaryTransformer(WAVELET_WIDTH), SpectrogramSummaryTransformer(
#     sample_rate= sample_rate, fft_length=200, stride_length=100)])


# In[ ]:


WINDOW_SIZE = 10000

# In[ ]:


STEP_SIZE = 5000

# In[ ]:


WAVELET_WIDTH = 40
