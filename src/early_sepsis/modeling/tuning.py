from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from early_sepsis.data.ingestion import DatasetFormat, load_dataset
from early_sepsis.data.preprocessing import build_preprocessor, prepare_training_data
from early_sepsis.modeling.evaluate import evaluate_binary_classifier
from early_sepsis.settings import AppSettings, get_settings


def tune_logistic_regression(
    data_path: str | Path,
    dataset_format: DatasetFormat = "auto",
    target_column: str | None = None,
    n_trials: int = 20,
    settings: AppSettings | None = None,
) -> dict[str, Any]:
    """Runs Optuna search over baseline logistic regression hyperparameters."""

    resolved_settings = settings or get_settings()
    resolved_target_column = target_column or resolved_settings.default_target_column

    dataframe = load_dataset(data_path=data_path, dataset_format=dataset_format)
    prepared = prepare_training_data(
        dataframe=dataframe,
        target_column=resolved_target_column,
        test_size=resolved_settings.train_test_split_ratio,
        random_state=resolved_settings.random_seed,
    )
    preprocessor = build_preprocessor(prepared.X_train)

    def objective(trial: optuna.Trial) -> float:
        c_value = trial.suggest_float("C", 1e-3, 20.0, log=True)
        max_iter = trial.suggest_int("max_iter", 200, 1000)

        classifier = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear",
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", classifier)])
        pipeline.fit(prepared.X_train, prepared.y_train)

        predictions = pipeline.predict(prepared.X_test)
        probabilities = pipeline.predict_proba(prepared.X_test)[:, 1]
        metrics = evaluate_binary_classifier(prepared.y_test, predictions, probabilities)
        return metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "trial_count": len(study.trials),
    }
