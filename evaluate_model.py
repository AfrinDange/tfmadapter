import argparse
import csv
import h5py
import json
import os

from dotenv import load_dotenv
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    RMSE,
    NRMSE,
)
from gluonts.model import evaluate_model as evaluate_model
from uni2ts.eval_util.evaluation import evaluate_model as evaluate_moirai
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset

from uni2ts.model.moirai import MoiraiForecast, MoiraiAdaptorForecast, MoiraiModule
from chronos import ChronosPredictor, ChronosAdaptorPredictor

import sys
sys.path.append("/workspaces/TST/fev/gift-eval/timesfm/src/")
import timesfm
from timesfm import TimesFmPredictor, TimesFmAdaptorPredictor

from baseline.regressor_predictor import RegressorPredictor
from baseline.nn_predictor import NNPredictor

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_names", type=str) # space separated names of datasets
parser.add_argument("--model_config", type=str) 
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--test_run", action="store_true")
parser.add_argument("--dataset_config", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--patch_size", type=int, default=32)

parser.add_argument("--filter_outliers", action="store_true")
parser.add_argument("--context_length", type=int)
parser.add_argument("--smaller_context", type=int)
parser.add_argument("--smallest_history", type=int)
parser.add_argument("--rolling_window_size", type=int)
parser.add_argument("--prediction_length", type=int)
parser.add_argument("--windows", type=int)
parser.add_argument("--num_past_k", type=int)
parser.add_argument("--distance", type=int)
parser.add_argument("--use_fixed_history", action="store_true")
parser.add_argument("--use_positions", action="store_true")
parser.add_argument("--pos_dims", type=int)
parser.add_argument('--adaptor_method', nargs="+", type=str, required=True)
parser.add_argument("--validation_metric", type=str, default="mse")
parser.add_argument("--folds", type=int, required=True)
parser.add_argument("--remove_pseudo_forecast_generator", action="store_true")
parser.add_argument("--no_window_selection", action="store_true")
parser.add_argument("--features_for_selection", type=str, required=True)
parser.add_argument("--log_subdir", type=str, required=True)

metrics = [
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    RMSE(),
    NRMSE(),
]

dataset_properties_map = json.load(open("notebooks/dataset_properties.json"))

pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

def create_result_logger(csv_file_path):
    file_exists = os.path.isfile(csv_file_path)
    is_empty = not file_exists or os.path.getsize(csv_file_path) == 0

    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if is_empty:
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "domain",
                    "num_variates",
                    "history",
                    "forecast_horizon",
                    "smaller_context",
                    "num_past_k",
                    "rolling_window_size",
                    "use_fixed_history",
                    "windows",
                    "distance",
                    "use_positions",
                    "pos_dims",
                    "filter_outliers",
                ]
            )

def run_eval(ds_name, dataset, args, ds_config, use_covariates, save_dir):
    ds_key = ds_name.split("/")[0]
    print(f"Processing dataset: {ds_name}")
    
    if "/" in ds_name:
        ds_key = ds_name.split("/")[0]
        ds_freq = ds_name.split("/")[1]
        ds_key = ds_key.lower()
        ds_key = pretty_names.get(ds_key, ds_key)
    else:
        ds_key = ds_name.lower()
        ds_key = pretty_names.get(ds_key, ds_key)
        ds_freq = dataset_properties_map[ds_key]["frequency"]

    config = f"{ds_key}/{ds_freq}/{args.model_config}"
    season_length = get_seasonality(dataset.freq)

    #### chronos
    print(args.model_name, "moirai" in args.model_name)
    if "chronos" in args.model_name:
        if "univariate" in args.model_config:
            predictor = ChronosPredictor(
                model_path=args.model_name,
                num_samples=20,
                prediction_length=ds_config["prediction_length"],
                save_dir=save_dir,
                model_config=args.model_config,
                test_run=args.test_run,
                device_map="cuda",
            )
        else:
            assert use_covariates
            predictor = ChronosAdaptorPredictor(
                model_path=args.model_name,
                batch_size=args.batch_size,
                num_samples=20,
                prediction_length=ds_config["prediction_length"],
                save_dir=save_dir,
                seasonality=season_length,
                use_covariates=use_covariates,
                model_config=args.model_config,
                test_run=args.test_run,
                ds_config=ds_config,
                device_map="cuda",
            )
        res = evaluate_model(
            predictor,
            test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
            metrics=metrics,
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
    elif "moirai" in args.model_name:
        if "univariate" in args.model_config:
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(args.model_name),
                prediction_length=1,
                context_length=4000,
                patch_size=args.patch_size,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        elif "covariate" in args.model_config:
            use_covariates = True
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(args.model_name),
                prediction_length=1,
                context_length=4000,
                patch_size=args.patch_size,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
        else:
            assert use_covariates
            model = MoiraiAdaptorForecast(
                module=MoiraiModule.from_pretrained(args.model_name),
                prediction_length=1,
                context_length=4000,
                patch_size=args.patch_size,
                num_samples=100,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0
            )
        with model.hparams_context(
            prediction_length = prediction_length,
            target_dim = 1,
            feat_dynamic_real_dim = dataset.feat_dynamic_real_dim if use_covariates else 0,
            past_feat_dynamic_real_dim = 0,
            context_length = context_length,
            patch_size = args.patch_size,
            num_samples = 100,
        ) as model: 
            model.hparams.context_length = ds_config["context_length"]
            model.hparams.prediction_length = ds_config["prediction_length"] #dataset.prediction_length
            model.hparams.target_dim = dataset.target_dim
            if use_covariates:
                model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim
                model.hparams.feat_dynamic_real_dim = dataset.feat_dynamic_real_dim
            predictor = model.create_predictor(
                batch_size=args.batch_size, 
                seasonality=season_length,
                use_covariates=use_covariates,
                save_dir=save_dir, 
                model_config=args.model_config, 
                test_run=args.test_run,
                ds_config=ds_config,
            )

            res = evaluate_moirai(
                predictor,
                test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
                metrics=metrics,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )
    elif "timesfm" in args.model_name:
        tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                num_layers=50,
                horizon_len=ds_config["prediction_length"],
                context_len=ds_config["context_length"],
                use_positional_embedding=False,
                output_patch_len=128,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=args.model_name),
        )
        if "univariate" in args.model_config:
            tfmpredictor = TimesFmPredictor(
                tfm=tfm,
                prediction_length=ds_config["prediction_length"],
                ds_freq=ds_freq,
                batch_size=args.batch_size
            )
        else:
            tfmpredictor = TimesFmAdaptorPredictor(
                tfm=tfm,
                prediction_length=ds_config["prediction_length"],
                ds_freq=ds_freq,
                batch_size=args.batch_size,
                seasonality=season_length,
                use_covariates=use_covariates,
                save_dir=save_dir,
                model_config=args.model_config,
                test_run=args.test_run,
                ds_config=ds_config
            )
        res = evaluate_model(
            tfmpredictor,
            test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
            metrics=metrics,
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length
        )
    elif "regression_baseline" in args.model_name:
        reg_predictor = RegressorPredictor(
            model_name=args.model_name,
            batch_size=args.batch_size,
            prediction_length=ds_config["prediction_length"],
            seasonality=season_length,
            use_covariates=use_covariates,
            save_dir=save_dir,
            model_config=args.model_config,
            test_run=args.test_run,
            ds_config=ds_config
        )
        res = evaluate_model(
            reg_predictor,
            test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
            metrics=metrics,
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length
        )
    elif "nbeatsx" in args.model_name.lower() or "tide" in args.model_name.lower():
        nn_predictor = NNPredictor(
            model_name=args.model_name,
            batch_size=args.batch_size,
            prediction_length=ds_config["prediction_length"],
            seasonality=season_length,
            use_covariates=use_covariates,
            save_dir=save_dir,
            model_config=args.model_config,
            test_run=args.test_run,
            ds_config=ds_config
        )
        res = evaluate_model(
            nn_predictor,
            test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
            metrics=metrics,
            batch_size=args.batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length
        )
    else:
        raise ValueError(f"{args.model_name} is unknown!")

    # Append the results to the CSV file
    if not args.test_run:
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    config,
                    args.model_name,
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                    ds_config["context_length"],
                    ds_config["prediction_length"],
                    ds_config["smaller_context"],
                    ds_config["num_past_k"],
                    ds_config["rolling_window_size"],
                    ds_config["use_fixed_history"],
                    ds_config["windows"],
                    ds_config["distance"],
                    ds_config["use_positions"],
                    ds_config["pos_dims"],
                    ds_config["filter_outliers"],
                ]
            )

        print(f'{args.model_config}:\tMAE={res["MAE[0.5]"][0]}, sMAPE={res["MAPE[0.5]"][0]}, RMSE={res["RMSE[mean]"][0]}')
        print(f"Results for {ds_name} have been written to {csv_file_path}")
    else:
        print(f'{args.model_config}:\tMAE={res["MAE[0.5]"][0]}, sMAPE={res["MAPE[0.5]"][0]}, RMSE={res["RMSE[mean]"][0]}')

if __name__ == "__main__":
    args = parser.parse_args()

    # list of eval datasets
    all_datasets = list(set(args.dataset_names.split()))
    ds_config=json.load(open(f"{args.dataset_config}.json"))

    for config in ["filter_outliers", "context_length", "smaller_context", "smallest_history", "rolling_window_size", "prediction_length", "windows", "num_past_k", "distance", "use_fixed_history", "use_positions", "pos_dims"]:
        if getattr(args, config, None) is not None:
            for ds_name in all_datasets:
                print(f"Overriding {config} for {ds_name} to {getattr(args, config, None)}")
                ds_config[ds_name][config] = getattr(args, config, None)
    
    use_covariates="with_cov" in args.model_config

    if use_covariates: 
        for ds_name in all_datasets:
            ds_config[ds_name]["adaptor_method"] = args.adaptor_method
            ds_config[ds_name]["validation_metric"] = args.validation_metric
            ds_config[ds_name]["folds"] = args.folds
            ds_config[ds_name]["remove_pseudo_forecast_generator"] = args.remove_pseudo_forecast_generator
            ds_config[ds_name]["no_window_selection"] = args.no_window_selection
            ds_config[ds_name]["features_for_selection"] = [features.split(",") for features in args.features_for_selection.split(";")]
            

    print(all_datasets)

    if "chronos" in args.model_name:
        output_dir = "/workspaces/TST/experiments/gifteval_results_/chronos/"
    elif "moirai" in args.model_name:
        output_dir = "/workspaces/TST/experiments/gifteval_results_/moirai/"
    elif "timesfm" in args.model_name:
        output_dir = "/workspaces/TST/experiments/gifteval_results_/timesfm/"
    elif "baseline" in args.model_name:
        output_dir = "/workspaces/TST/experiments/gifteval_results_/baseline"
    else:
        raise ValueError(f"{args.model_name} unknown!")
    
    if not args.test_run:
        # open writer to log results
        if args.log_subdir != "":
            output_dir = os.path.join(output_dir, args.log_subdir)
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, f"results.csv")
        create_result_logger(csv_file_path)
    
    # iterate over datasets
    for ds_name in all_datasets:
        print(type(ds_config[ds_name]), ds_config[ds_name])
        dataset = Dataset(name=ds_name, to_univariate=ds_config[ds_name]["to_univariate"])

        context_length = ds_config[ds_name]["context_length"]
        prediction_length = ds_config[ds_name]["prediction_length"]

        # check if results already saved
        results_dir=os.path.join(output_dir, f"{ds_name}_h_{context_length}_f_{prediction_length}.h5")
        if not args.test_run:
            with h5py.File(results_dir, "a") as f:
                if args.model_config in f:
                    raise ValueError(f"{args.model_config} already saved!")
                else:
                    f.create_group(args.model_config)

        run_eval(ds_name, dataset, args, ds_config[ds_name], use_covariates=use_covariates, save_dir=results_dir)