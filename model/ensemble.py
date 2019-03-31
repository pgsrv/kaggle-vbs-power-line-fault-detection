import optuna
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_validate


class Cutoff(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y, **kwargs):
        return self

    def predict(self, X, **kwargs):
        return (X >= self.threshold).astype("int")

    def get_params(self, *args, **kwargs):
        return {"threshold": self.threshold}


class CutOffOptimizer(object):

    def objective(self, trial: optuna.trial.Trial):
        threshold = trial.suggest_uniform('threshold', 0, 1)
        avg = self.x.mean(axis=1)
        scores = cross_validate(Cutoff(threshold), avg, self.y, verbose=1,
                                n_jobs=-1, scoring=make_scorer(matthews_corrcoef), cv=5)
        return - scores["test_score"].mean()

    def optimize(self, x, y):
        self.x, self.y = x, y
        study = optuna.create_study()

        study.optimize(self.objective, n_trials=100)
        print("best score: {}".format(study.best_value))
        print("best params: {}".format(study.best_params))

        best_model = Cutoff(**study.best_params)
        return best_model

# class CutOffFromKfoldResult(object):
#
#     def __init__(self, model_path_root: Path):
#
#         self.model_path_root = model_path_root
#
#         self._current_fold = 0
#
#
#     def predict(self, dataset: [TorchSimpleDataset, Feature], batch_size=256, suffix="", measurement_ids=None):
#         folds = [int(path.name.replace("cv", "")) for path in self.model_path_root.glob("cv*")]
#         self._fold_results = []
#         self.suffix = suffix
#         if measurement_ids is not None:
#             self._measure_ids = measurement_ids
#         else:
#             self._measure_ids = dataset.measurement_ids
#
#         for fold in folds:
#             self._current_fold = fold
#             self._predict_fold(dataset, batch_size)
#
#         self._fold_df = pd.DataFrame()
#         self._fold_df["id_measurement"] = self._measure_ids
#         self._fold_df["hard_voted_class"] = self._hard_vote(self._fold_results)
#         self._fold_df["avg_probability"], self._fold_df["soft_voted_class"] = self._soft_vote(self._fold_results)
#         self._fold_df.to_csv(self.model_path_root.joinpath(suffix + "predicted.csv"))
#         self._fold_results.append(self._fold_df)
#
#         self._fold_df.to_csv(self.model_path_root.joinpath(suffix + "predicted.csv"))
#         return self._fold_df
#
#     def _predict_fold(self, dataset: [TorchSimpleDataset, Feature], batch_size=256):
#         self._current_fold_root = self.model_path_root.joinpath("cv{}".format(self._current_fold))
#
#         logger.info("**** predicting with fold {} ****".format(self._current_fold))
#         self.load_model()
#         if isinstance(dataset, Feature):
#             predicted = np.concatenate(
#                 [self._model.predict(TorchSimpleDataset(x, self._measure_ids), batch_size, n_cpus)
#                  for x in dataset.partial_load()], axis=0)
#             print(predicted.shape)
#             dataset.flash()
#         else:
#             predicted = self._model.predict(dataset, batch_size, n_cpus)
#         df = pd.DataFrame()
#         df["id_measurement"] = self._measure_ids
#         print(df.shape)
#         df["probability"] = predicted
#         df["class"] = binarize(predicted, threshold=self.threshold)
#         df.to_csv(self._current_fold_root.joinpath(self.suffix + "predicted.csv"))
#         self._fold_results.append(df)
#
#     @abstractmethod
#     def create_model(self, model_path):
#         pass
#
#     @staticmethod
#     def to_submission(df):
#         logger.info("creating submission form")
#         test_df = VbsDataSetFactory().test_df
#         soft_df = df[["id_measurement", "soft_voted_class"]]
#         hard_df = df[["id_measurement", "hard_voted_class"]]
#
#         def f(df: pd.DataFrame):
#             submission_df = test_df.merge(df, on="id_measurement", how="left")
#             submission_df.drop(["id_measurement", "phase"], axis=1, inplace=True)
#             col = [str(col) for col in submission_df.columns if str(col) != "signal_id"][0]
#             submission_df.rename(columns={col: "target"}, inplace=True)
#             submission_df["target"] = submission_df["target"].astype(bool)
#             return submission_df
#
#         submission_dfs = {}
#         submission_dfs["soft"] = f(soft_df)
#         submission_dfs["hard"] = f(hard_df)
#
#         return submission_dfs
