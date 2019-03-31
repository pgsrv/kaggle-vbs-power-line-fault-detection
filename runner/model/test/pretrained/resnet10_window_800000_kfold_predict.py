from pathlib import Path

from model.nn_model_wrapper import DefaultTransformers
from runner.model.plot_model_runner import ResNet10PlotKfoldModel

if __name__ == '__main__':
    plot_root = Path(__file__).parent.parent.parent.parent.joinpath(
        "/mnt/share/vbs-power-line-fault-detection/features/test/window_800000_stride_800000/")
    model_root = Path(
        "/mnt/share/vbs-power-line-fault-detection/models/resnet10_kfold/window_800000_stride_800000")

    model = ResNet10PlotKfoldModel(model_path_root=model_root,
                                   dropout_rate=0.5, threshold=0.5)

    df = model.predict(plot_root, batch_size=1600, transformers=DefaultTransformers())
    submission_dfs = model.to_submission(df)
    for key, submission_df in submission_dfs.items():
        submission_df.to_csv(model_root.joinpath("submission_{}.csv".format(key)), index=None)

    print("done!!")
