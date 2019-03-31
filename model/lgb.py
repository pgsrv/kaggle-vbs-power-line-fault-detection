import logging

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import binarize

from feature.base import RANDOM_STATE, Features
from utils.data.dataset import VbsDataSet

logger = logging.getLogger(__name__)


class LightGBMWrapper(object):

    def train(self, feature: Features, vbs_dataset: VbsDataSet, save_dir, n_trials=100, cv=10):
        self._X = feature.get_values()
        self._Y = np.asarray(vbs_dataset.targets)
        self.cv = cv
        self.study = optuna.create_study()

        self.study.optimize(self.objective, n_trials=n_trials)

        logger.info("best score: {}\nbest params: {}".format(self.study.best_value, str(self.study.best_params)))

        self.study.trials_dataframe().to_json(save_dir.joinpath("study.json"), force_ascii=False, orient='records')
        best_params = self.study.best_params
        # In[ ]:

        best_params["random_state"] = RANDOM_STATE

        # In[ ]:

        clf = LGBMClassifier(**best_params)

        # In[ ]:

        clf.fit(self._X, self._Y, eval_metric=matthews_corrcoef, verbose=1)

        joblib.dump(clf, str(save_dir.joinpath("model")))

    def objective(self, trial: optuna.trial.Trial):
        boosting_type = trial.suggest_categorical("boosting_type", ['gbdt', 'dart'])
        num_leaves = trial.suggest_int('num_leaves', 30, 80)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 10, 100)
        #     max_depth = trial.suggest_int('max_depth', )
        lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-5, 1e-2)
        lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-5, 1e-2)
        #     num_iterations = trial.suggest_int("num_iterations", 100, 500)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

        clf = LGBMClassifier(boosting_type=boosting_type, num_leaves=num_leaves,
                             learning_rate=learning_rate, reg_alpha=lambda_l1,
                             min_child_samples=min_data_in_leaf,
                             reg_lambda=lambda_l2, random_state=RANDOM_STATE)
        #     fit_params = {"early_stopping_rounds":20,
        #                  "eval_metric": matthews_corrcoef}
        scores = cross_validate(clf, self._X, self._Y, verbose=1,
                                n_jobs=-1, scoring=make_scorer(matthews_corrcoef), cv=self.cv)
        return - scores["test_score"].mean()

    def load(self, model_path):
        self.model = joblib.load(model_path)
        return self

    def predict(self, feature: Features, vbs_dataset: VbsDataSet, save_dir, threshold=0.5,
                submission=True):
        self._X = feature.get_values()

        predicted = self.model.predict \
            (self._X)
        self._vbs_dataset = vbs_dataset

        predicted_df = pd.DataFrame()
        predicted_df["id_measurement"] = vbs_dataset.measurement_ids
        predicted_df["probability"] = predicted
        predicted_df["target"] = binarize(predicted.reshape((-1, 1)), threshold=threshold)
        predicted_df.to_csv(save_dir.joinpath("predicted.csv"), index=None)

        if not submission:
            return predicted_df

        submission_df = self.to_submission(predicted_df.drop("probability", axis=1),
                                           vbs_dataset.meta_df)
        submission_df.to_csv(save_dir.joinpath("submission.csv"), index=None)

        return submission_df

    @staticmethod
    def to_submission(df, source_df):
        submission_df = source_df.merge(df, on="id_measurement", how="left")
        submission_df.drop(["id_measurement", "phase"], axis=1, inplace=True)
        col = [str(col) for col in submission_df.columns if str(col) != "signal_id"][0]
        submission_df.rename(columns={col: "target"}, inplace=True)
        submission_df["target"] = submission_df["target"].astype(bool)
        return submission_df
