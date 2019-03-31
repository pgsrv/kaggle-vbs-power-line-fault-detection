from feature.base import WindowFeature, WindowStandardScaled


def run(feature_root):
    dummy_window = 200
    dummy_size = 200
    src_feature = WindowFeature(None, dummy_window, dummy_size, accept_window=True)
    # feature_root = Path(
    #     "/mnt/share/vbs-power-line-fault-detection/features/fft/"
    #     "fft_length_{}_stride_{}".format(
    #         fft_length, fft_stride))
    save_root = feature_root.joinpath("normalized")
    save_root.mkdir(exist_ok=True, parents=True)

    scaled_feature = WindowStandardScaled(dummy_size, dummy_window, accept_window=True)
    print("writing train data ....")
    train_src_path = feature_root.joinpath("train/feature.pickle")
    train_save_root = save_root.joinpath("train")

    train_save_root.mkdir(exist_ok=True, parents=True)
    src_feature.load(train_src_path)
    scaled_feature.fit_transform(src_feature, train_save_root, n_jobs=-1)

    print("writing test data....")
    test_paths = list(feature_root.joinpath("test").glob("feature*.pickle"))
    src_feature.load(test_paths)
    test_save_root = save_root.joinpath("test")

    test_save_root.mkdir(exist_ok=True, parents=True)
    scaled_feature.transform(src_feature, test_save_root, n_jobs=-1)

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
