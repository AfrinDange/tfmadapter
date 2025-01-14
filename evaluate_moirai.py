import argparse
import csv
import h5py
import json
import os

from dotenv import load_dotenv
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MSE,
    MAE,
    MASE,
    MAPE,
    SMAPE,
    MSIS,
    RMSE,
    NRMSE,
    ND,
    MeanWeightedSumQuantileLoss
)
# from gluonts.model import evaluate_model
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiAdaptorForecast, MoiraiAdaptorExtendedForecast, MoiraiModule

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_names", type=str) # space separated names of datasets
parser.add_argument("--model_config", type=str) #, choices=["moirai_no_cov", "moirai_with_cov", "kernel_regr_with_cov", "ensemble_with_cov"])
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--test_run", action="store_true")
parser.add_argument("--dataset_config", type=str, required=True)
parser.add_argument("--patch_size", type=int, default=32)
parser.add_argument("--model_name", type=str, required=True)

metrics = [
    # MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    # MSIS(),
    RMSE(),
    NRMSE(),
    # ND(),
    # MeanWeightedSumQuantileLoss(
    #     quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # ),
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
                    # "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    # "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    # "eval_metrics/ND[0.5]",
                    # "eval_metrics/mean_weighted_sum_quantile_loss",
                    "domain",
                    "num_variates",
                    "history",
                    "forecast_horizon",
                    "smaller_context",
                    "num_past_k",
                    "rolling_window_size",
                    "use_fixed_history",
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

    # set the Moirai hyperparameter according to each dataset, then create the predictor
    model.hparams.context_length = ds_config["context_length"]
    model.hparams.prediction_length = ds_config["prediction_length"] #dataset.prediction_length
    model.hparams.target_dim = dataset.target_dim
    if use_covariates:
        model.hparams.past_feat_dynamic_real_dim = dataset.past_feat_dynamic_real_dim
        model.hparams.feat_dynamic_real_dim = dataset.feat_dynamic_real_dim

    predictor = model.create_predictor(
        batch_size=args.batch_size, 
        save_dir=save_dir, 
        model_config=args.model_config, 
        test_run=args.test_run,
        ds_config=ds_config,
    )

    season_length = get_seasonality(dataset.freq)

    res = evaluate_model(
        predictor,
        test_data=dataset.custom_test_data(context_length=ds_config["context_length"], prediction_length=ds_config["prediction_length"], windows=ds_config["windows"], distance=ds_config["distance"]),
        metrics=metrics,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    # Append the results to the CSV file
    if not args.test_run:
        with open(csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    config,
                    "moirai_small",
                    # res["MSE[mean]"][0],
                    res["MSE[0.5]"][0],
                    res["MAE[0.5]"][0],
                    res["MASE[0.5]"][0],
                    res["MAPE[0.5]"][0],
                    res["sMAPE[0.5]"][0],
                    # res["MSIS"][0],
                    res["RMSE[mean]"][0],
                    res["NRMSE[mean]"][0],
                    # res["ND[0.5]"][0],
                    # res["mean_weighted_sum_quantile_loss"][0],
                    dataset_properties_map[ds_key]["domain"],
                    dataset_properties_map[ds_key]["num_variates"],
                    ds_config["context_length"],
                    ds_config["prediction_length"],
                    ds_config["smaller_context"],
                    ds_config["num_past_k"],
                    ds_config["rolling_window_size"],
                    ds_config["use_fixed_history"],
                ]
            )

        print(f'{args.model_config}:\tMAE={res["MAE[0.5]"][0]}, MAPE={res["MAPE[0.5]"][0]}')
        print(f"Results for {ds_name} have been written to {csv_file_path}")
    else:
        print(f'{args.model_config}:\tMAE={res["MAE[0.5]"][0]}, MAPE={res["MAPE[0.5]"][0]}')

if __name__ == "__main__":
    args = parser.parse_args()

    # list of eval datasets
    all_datasets = list(set(args.dataset_names.split()))
    ds_config=json.load(open(f"{args.dataset_config}.json"))

    use_covariates="with_cov" in args.model_config

    print(all_datasets)

    output_dir = "/workspaces/TST/experiments/gifteval_results/moirai/"
    if not args.test_run:
        # open writer to log results
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, f"results.csv")
        create_result_logger(csv_file_path)

    # load the model
    if "moirai" in args.model_config:
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
    elif "extended" in args.model_config:
        model=MoiraiAdaptorExtendedForecast(
            module=MoiraiModule.from_pretrained(args.model_name),
            prediction_length=1,
            context_length=4000,
            patch_size=args.patch_size,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0
        )
    else:
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

        with model.hparams_context(
            prediction_length = prediction_length,
            target_dim = 1,
            feat_dynamic_real_dim = dataset.feat_dynamic_real_dim if use_covariates else 0,
            past_feat_dynamic_real_dim = 0,
            context_length = context_length,
            patch_size = args.patch_size,
            num_samples = 100,
        ) as model:
            run_eval(ds_name, dataset, args, ds_config[ds_name], use_covariates=use_covariates, save_dir=results_dir)