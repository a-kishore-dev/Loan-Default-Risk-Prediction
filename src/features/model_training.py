import os
import mlflow
import warnings
import shap
import polars as pl
import matplotlib.pyplot as plt
from pydantic import BaseModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score
from src.features.utils import save_object

warnings.filterwarnings("ignore")

mlflow.set_experiment(experiment_name="Credit Risk")
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")


class ModelTrainerConfig(BaseModel):
    model_path: str = os.path.join("models","model.pkl")
    table_path: str = os.path.join("artifact","table.parquet")

class ModelTrainer:
    
    def __init__(self):
        self.model_config = ModelTrainerConfig()
    
    def evaluate_model(self, model, param, X_train, y_train, X_test, y_test):
        '''
        Evaluate the model with different parameters and return the report
        '''
        res = {}
        mlflow.sklearn.autolog(log_models=False)
        for i in range(len(model)):
            curr_model_name, curr_model = list(model.items())[i]
            curr_model_params = param[curr_model_name]
            with mlflow.start_run(run_name=curr_model_name):
                r_cv = RandomizedSearchCV(curr_model, curr_model_params, cv=2, n_jobs=-1, scoring=["roc_auc","f1"], refit="roc_auc", verbose=0)
                r_cv.fit(X_train, y_train)

                best_model = r_cv.best_estimator_

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Probabilities for ROC-AUC
                y_train_prob = best_model.predict_proba(X_train)[:, 1]
                y_test_prob = best_model.predict_proba(X_test)[:, 1]

                y_train_f1_score = f1_score(y_train, y_train_pred)
                y_test_f1_score = f1_score(y_test, y_test_pred)

                y_train_roc = roc_auc_score(y_train, y_train_prob)
                y_test_roc = roc_auc_score(y_test, y_test_prob)

                mlflow.sklearn.log_model(best_model, artifact_path="Best_" + curr_model_name, input_example=X_train[[0]])

                mlflow.log_metric("testing_roc_auc ", y_test_roc)

                res[curr_model_name] = {
                    "best_params": r_cv.best_params_,
                    "y_train_f1_score": y_train_f1_score,
                    "y_test_f1_score": y_test_f1_score,
                    "y_train_roc": y_train_roc,
                    "y_test_roc": y_test_roc,
                    "best_model": best_model
                }
        return res
    
    def create_comparison_table(self, report):
        '''
        Create a comparison tabel for understanding the model performance
        '''
        schema = {
            "model_name": str,
            "best_params": str,
            "y_train_f1_score": pl.Float64,
            "y_test_f1_score": pl.Float64,
            "y_train_roc": pl.Float64,
            "y_test_roc": pl.Float64,
            "best_model": str
        }
        df = pl.DataFrame(schema=schema)

        for key, value in report.items():
            new_row = [key] + [str(val) if not isinstance(val, float) else val for val in value.values()]
            new_row_df = pl.DataFrame([new_row], schema=df.schema, orient='row')
            df = df.vstack(new_row_df)
        
        df.write_parquet(self.model_config.table_path)
        return

    def initiate_model_trainer(self, train_arr, valid_arr, feature_names):
        '''
        Train the model with data and return the best model
        '''
        X_train, y_train, X_test, y_test = (
            train_arr[:,: -1],
            train_arr[:,-1],
            valid_arr[:,: -1],
            valid_arr[:,-1],
        )

        models = {
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "ada_boost": AdaBoostClassifier(),
            "xg_boost": XGBClassifier(),
            "lightgbm": LGBMClassifier(),
            "catboost": CatBoostClassifier()
        }

        params = {
            "logistic_regression": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["saga", "lbfgs"],
                "class_weight": [None, "balanced"],
                "max_iter": [500, 1000]
            },
            "random_forest": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
                "class_weight": ["balanced", "balanced_subsample"]
            },
            "ada_boost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.5]
            },
            "xg_boost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [11.39],
                "min_child_weight": [1, 3],
                "gamma": [0, 0.1]
            },
            "lightgbm": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [-1, 5, 10],
                "num_leaves": [31, 50, 100],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "scale_pos_weight": [11.39],
                "min_child_samples": [10, 20]
            },
            "catboost": {
                "iterations": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "depth": [4, 6, 8],
                "l2_leaf_reg": [1, 3, 5],
                "scale_pos_weight": [11.39]
            }
        }

        model_report = self.evaluate_model(models, params, X_train, y_train, X_test, y_test)
        print(model_report)
        self.create_comparison_table(model_report)
        model_report_sorted = sorted(model_report.items(), key=lambda x: x[1]["y_test_roc"], reverse=True)

        best_model = model_report_sorted[0][1]['best_model']

        explainer = shap.Explainer(best_model, X_test)
        shap_values = explainer(X_test)

        # Waterfall plot — single prediction with proper feature names
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0].values,
                base_values=shap_values[0].base_values,
                data=X_test[0],
                feature_names=feature_names,
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig("artifact/shap_waterfall_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Summary plot with feature names
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("artifact/shap_summary_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

        
        save_object(
            best_model,
            file_path=self.model_config.model_path
        )

        return best_model

