from pathlib import Path

from runner.model.plot_model_runner import ResNet18PlotKfoldModel

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "output/features/test/window_800000_stride_800000/")
    model_root = Path(
        "/mnt/gcs/kaggle-vbs-power-line-fault-detection/models/resnet18_pretrained_kfold/window_800000_stride_800000")

    avg_model = ResNet18PlotKfoldModel(model_path_root=model_root,
                                       dropout_rate=0.5, threshold=0.5)

    df = avg_model.predict(plot_root, batch_size=800)

    submission_dfs = avg_model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(model_root.joinpath("submission_{}.csv".format(key)), index=None)
    print("done!!")