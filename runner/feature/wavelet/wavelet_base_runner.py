from pathlib import Path

from feature.base import WaveletFeature
from utils.data import dataset


def run(wavelet_width):
    feature = WaveletFeature(wavelet_width)
    root = Path(
        "/mnt/share/vbs-power-line-fault-detection/features/wavelet/"
        "width_{}_mexh".format(wavelet_width))

    root.mkdir(exist_ok=True, parents=True)
    print("writing train wavelet  data ....")
    train_path = root.joinpath("train")
    train_path.mkdir(exist_ok=True, parents=True)
    train_data_set = dataset.VbsDataSetFactory()(is_train=True)
    feature.transform(train_data_set, train_path)

    print("writing test wavelet data....")
    test_data_set = dataset.VbsDataSetFactory()(is_train=False)
    test_path = root.joinpath("test")
    test_path.mkdir(exist_ok=True, parents=True)
    feature.transform(test_data_set, test_path)

    print("done!")
#
#
# def test_run(window_size, step_size, fft_length, fft_stride):
#     feature = FftSummaryFeature(window_size=window_size, step_size=step_size,
#                                 fft_length=fft_length, fft_stride=fft_stride)
#     root = Path(
#         "/mnt/share/vbs-power-line-fault-detection/features/fft/"
#         "window_{}_step_{}/fft_length_{}_stride_{}".format(
#             window_size,
#             step_size, fft_length, fft_stride))
#
#     root.mkdir(exist_ok=True, parents=True)
# print("writing train fft summary data ....")
# train_path = root.joinpath("train")
# train_path.mkdir(exist_ok=True, parents=True)
# train_data_set = dataset.VbsDataSetFactory()(is_train=True)
# # feature.transform(train_data_set, train_path)
#
# print("writing test fft summary data....")
# test_data_set = dataset.VbsDataSetFactory()(is_train=False)
# test_path = root.joinpath("test")
# test_path.mkdir(exist_ok=True, parents=True)
# feature.transform(test_data_set, test_path)
#
# print("done!")
