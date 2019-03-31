import itertools

from utils.data.dataset import VbsDataSetFactory, TorchSimpleDataset


def run(feature, saved_feature_root, params, model_class, save_dir, batch_size=128, padded=False):
    # saved_feature_root = Path("/mnt/share/vbs-power-line-fault-detection/features/summary/"
    #                           "window_{}_step_{}".format(40, 20))
    # save_dir = Path(
    #     "/mnt/share/vbs-power-line-fault-detection/models/hierarchical_gru/summary/window_{}_step_{}".format(200, 200))
    SWITCH_INDICES = list(itertools.chain.from_iterable([[i, i + 5, i + 10] for i in range(5)]))
    save_dir.mkdir(parents=True, exist_ok=True)
    # feature = SummaryWindowFeature(40, 20)
    print("loading data")
    if padded:
        feature.load(list(sorted(saved_feature_root.glob("test/feature*.picklepadded.pickle"))))
    else:
        feature.load(list(sorted(saved_feature_root.glob("test/feature*.pickle"))),
                     SWITCH_INDICES)
    # X = feature.get_all_rows()
    #
    # print(X.shape)
    train_dataset = VbsDataSetFactory()(is_train=False)
    # params = HierarchicalGruParams(
    #     seq_len=200,
    #     chunk_len=200,
    #     input_size=15,
    #     hidden_sizes=(128, 128),
    #     gru_dropout=0,
    #     dense_output=64,
    #     dropout=0.3,
    #     num_layers=1,
    #     pool_size=None,
    #     pool_stride=None
    # )
    # test_dataset = VbsDataSetFactory()(is_train=False)
    # test_dataset = TorchSimpleDataset(X, test_dataset.measurement_ids)
    model = model_class(params, model_path_root=save_dir, threshold=0.5)
    df = model.predict(feature, batch_size=batch_size, measurement_ids=train_dataset.measurement_ids)
    submission_dfs = model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(save_dir.joinpath("submission_{}.csv".format(key)), index=None)

    feature.load(saved_feature_root.joinpath("train/feature.pickle"), SWITCH_INDICES)
    X = feature.feature_array
    test_dataset = VbsDataSetFactory()(is_train=True)
    test_dataset = TorchSimpleDataset(feature.feature_array, test_dataset.measurement_ids)

    df = model.predict(test_dataset, batch_size=batch_size, suffix="train")

    print("done!!")
