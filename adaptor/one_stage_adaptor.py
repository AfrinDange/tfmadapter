import csv
import cupy
import os
import re
import torch

import optuna
from optuna.samplers import TPESampler

from cuml import KernelRidge
from dataclasses import dataclass
from jaxtyping import Float, Int
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from skopt import gp_minimize
from skopt.space import Real
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
from xgboost import XGBRegressor

@dataclass
class DataClassConfig:
    name: str
    context_length: int
    smaller_context: int
    smallest_history: int
    rolling_window_size: int 
    prediction_length: int
    windows: int
    distance: int
    num_past_k: int
    to_univariate: bool
    use_fixed_history: bool
    use_positions: bool
    pos_dims: int
    filter_outliers: bool
    run_name: bool
    adaptor_method: List[str]
    features_for_selection: List[List[str]]
    validation_metric: str
    folds: int
    remove_pseudo_forecast_generator: bool
    no_window_selection: bool

VALIDATE_MODELS = {
    0: {
        "model": "gaussian_process",
        "kernel_type": "rbf",
        "threshold": [2.0, 2.5, 3.0],
    },
    1: {
        "model": "gaussian_process",
        "kernel_type": "matern",
        "threshold": [2.0, 2.5, 3.0],
    },
    2: {
        "model": "kernel_comp_gaussian_process",
        "threshold": [2.0, 2.5, 3.0],
    },
    3: {
        "model": "bayesian"
    }

}

def mae(
    y_true: Float[torch.Tensor, "batch num_test_examples patch_size"], 
    y_pred: Float[torch.Tensor, "batch num_test_examples patch_size"]
    ) -> Float[torch.Tensor, "batch patch_size"]:
    y_true = torch.flatten(y_true, start_dim=1)
    y_pred = torch.flatten(y_pred, start_dim=1)

    return torch.mean(torch.abs(y_true - y_pred), dim=1) 

def mse(
    y_true: Float[torch.Tensor, "batch num_test_examples patch_size"], 
    y_pred: Float[torch.Tensor, "batch num_test_examples patch_size"]
    ) -> Float[torch.Tensor, "batch patch_size"]:
    y_true = torch.flatten(y_true, start_dim=1)
    y_pred = torch.flatten(y_pred, start_dim=1)

    return torch.mean(torch.abs(y_true - y_pred)**2, dim=1) 

class OneStageAdaptor:
    ''''
        
    '''
    def __init__(
        self,
        fm_predictions_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        fm_predictions_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_true: Float[torch.Tensor, "batch num_train_examples patch_size"],
        y_mean: Float[torch.Tensor, "batch 1 1"],
        y_std: Float[torch.Tensor, "batch 1 1"],
        cov_mean: Float[torch.Tensor, "batch 1 num_covariates"],
        cov_std: Float[torch.Tensor, "batch 1 num_covariates"],
        features: Dict[str, Float[torch.Tensor, "batch num_total_examples feature_dim"]],
        ds_config: Dict[str, Any],
        future_target: Optional[Float[torch.Tensor, "batch pred_length patch_size"]] = None,
    ):
        '''
            features:
                # with unused history
                    - covariates, positions, past_k

                # with entire history
                    - all_covariates, all_positions, all_history
        
        '''
        self.ds_config = DataClassConfig(**ds_config)
        self.entire_context_length = features["all_history"].shape[1]
        self.prediction_length = self.ds_config.prediction_length
        self.num_covariates = features["covariates"].shape[-1]
        self.num_past_k = features["past_k"].shape[-1]
        self.num_pos = features["all_positions"].shape[-1]
        self.validation_metric = mse if self.ds_config.validation_metric == "mse" else mae

        # normalization params
        self.eps = 1e-6
        self.y_mean = y_mean
        self.y_std = y_std + self.eps
        self.cov_mean = cov_mean
        self.cov_std = cov_std + self.eps 

        self.fm_predictions_train = fm_predictions_train
        self.fm_predictions_test = fm_predictions_test
        self.y_true = y_true

        self.future_target = future_target

        # features and config
        self.features = features

        features["all_past_k"] = torch.cat([
                features["all_history"][:,:1].repeat(1, self.num_past_k, 1),
                features["all_history"],
                self.fm_predictions_test[:, :-1]
            ], dim=1).unfold(dimension=1, size=self.num_past_k, step=1).squeeze(-2)
        
        assert features["all_past_k"].shape[1] == self.entire_context_length + self.prediction_length, f"{features['all_past_k'].shape} current shape"

        # positions 
        features["all_positions"] = features["all_positions"].unsqueeze(0).repeat(self.y_true.shape[0], 1, 1)

        self.validate_features = {id:{"features": features} for id, features in enumerate(self.ds_config.features_for_selection)}

        # evaluate adaptor methods
        self.validate_models = {}
        model_idx = 0
        self.gp_id = []
        for i, adaptor_params in VALIDATE_MODELS.items():
            if adaptor_params["model"] in self.ds_config.adaptor_method:
                self.validate_models[model_idx] = adaptor_params
                if "gaussian_process" in adaptor_params["model"]:
                    self.gp_id.append(model_idx)
                model_idx += 1
                print(f"Adding {adaptor_params['model']} to model selection. Configuration {adaptor_params}")
        # assert len(self.ds_config.adaptor_method) == len(self.validate_models)

        # gp opt 

        self.gp_opt_params = {}
        for id in self.gp_id:
            self.gp_opt_params[id] = {
                k: {} for k in self.validate_features
            }

        self.nadaraya_watson_opt_params = {
            k: {} for k in self.validate_features
        }
        
    def _cross_validate(
        self
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            Cross validate
                * adaptor method
                * features

            Train the final model with all training data
        '''
        eval_metrics = dict()
        num_examples = self.x_train.shape[1]

        # create cross validation sets
        folds=self.ds_config.folds
        for i in tqdm(range(folds, 0, -1), desc=f"{folds}-fold validation", leave=False):
            train_length = num_examples - (i * self.prediction_length)
            x_train = self.x_train[:, :train_length].clone().detach()
            y_train = self.y_train[:, :train_length].clone().detach()

            x_val = self.x_train[:, train_length:train_length+self.prediction_length].clone().detach()
            y_val = self.y_train[:, train_length:train_length+self.prediction_length].clone().detach()

            # returns a dictionary "method": [loss x 5 folds]
            self.evaluate_adaptors(x_train, y_train, x_val, y_val, eval_metrics)  

              
        # test prediction with best method per model
        all_error=[]
        for method, error in eval_metrics.items():
            # method codestring error List[Float[torch.tensor, "batch"]]
            error = torch.stack(error).mean(dim=0)
            all_error.append(error)

        error_per_method = torch.stack(all_error) # methods x bsz
        best_method_per_example = torch.argmin(error_per_method, dim=0) # bsz

        print(best_method_per_example)
        
        # best_method = torch.argmax(torch.bincount(best_method_per_example)).item()

        # create a csv with list(eval_metrics.keys()) as columns and add for each example the method's loss
        # Create CSV file with method names as columns
        try:
            csv_file = f'/workspaces/TST/fev/gift-eval/cross_validate/{self.ds_config.run_name}_{self.ds_config.name}.csv'
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header if file does not exist
                if not file_exists:
                    header = ['example'] + list(eval_metrics.keys())
                    writer.writerow(header)
                else:
                    with open(csv_file, mode='r') as read_file:
                        reader = csv.reader(read_file)
                        existing_header = next(reader)
                        if len(existing_header) != len(['example'] + list(eval_metrics.keys())):
                            raise ValueError("Number of columns in the existing file does not match the new data")
                
                # Write data
                for example_idx in range(error_per_method.shape[1]):
                    row = [example_idx] + error_per_method[:, example_idx].tolist()
                    writer.writerow(row)
        except: 
            print(f"Could not log cross validation error")

        # nadaraya watson
        for feature_type in self.nadaraya_watson_opt_params:
            if "width" in self.nadaraya_watson_opt_params[feature_type]:
                self.nadaraya_watson_opt_params[feature_type]["width"] = torch.stack(self.nadaraya_watson_opt_params[feature_type]["width"]).mean(dim=0)
            if "variance" in self.nadaraya_watson_opt_params[feature_type]:
                self.nadaraya_watson_opt_params[feature_type]["variance"] = torch.stack(self.nadaraya_watson_opt_params[feature_type]["variance"]).mean(dim=0)

        for adaptor_type in self.gp_opt_params:
            for feature_type in self.gp_opt_params[adaptor_type]:
                for param in self.gp_opt_params[adaptor_type][feature_type]:
                    self.gp_opt_params[adaptor_type][feature_type][param] = torch.stack(self.gp_opt_params[adaptor_type][feature_type][param]).mean(dim=0)
                
        return best_method_per_example, eval_metrics

    def predict(
        self, 
        x_train,
        y_train,
        x_test,
        best_method_per_example: Float[torch.Tensor, "bsz"]=None, 
        eval_metrics: Dict[str, List[Float[torch.Tensor, "batch"]]]=None,
    ) -> Float[torch.Tensor, "bsz prediction_length patch_size"]:
        batch_size = x_train.shape[0]

        batch_y_test = torch.empty((batch_size, self.prediction_length, 1), device=x_test.device)

        # def predict_per_example(example, x_train, y_train, x_test):
        for example in tqdm(range(batch_size), desc="Final Predictions: ", leave=False):
            best_method = list(eval_metrics)[best_method_per_example[example]]
            
            if any([f"adaptor_{str(id)}" in best_method for id in self.gp_id]):
                # find configuration
                best_adaptor, best_feature, threshold = re.match(r"adaptor_(\d+)_feature_(\d+)_threshold_(\d+)", best_method).groups()
                assert self.validate_models[int(best_adaptor)]["model"] == "gaussian_process"

                adaptor_params = self.validate_models[int(best_adaptor)]

                ux_train, ux_test, ux_mean, ux_std, uy_train, uy_mean, uy_std = self.preprocess_data(
                    example=example,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    feature_names=self.validate_features[int(best_feature)]["features"]
                )

                optimized_kernel_params = {param: batch_val[example] for param, batch_val in self.gp_opt_params[int(best_adaptor)][int(best_feature)].items()}

                y_test = self.eval_gaussian_process_estimator(
                    x_train=ux_train,
                    y_train=uy_train,
                    x_test=ux_test,
                    y_sub=self.fm_predictions_test[example].unsqueeze(0),
                    y_mean=uy_mean,
                    y_std=uy_std,
                    **{**adaptor_params, **optimized_kernel_params, "threshold": [] if int(threshold) == 0 else [self.validate_models[int(best_adaptor)]["threshold"][int(threshold)-1]]}
                )[-1] # select best method threshold / no threshold
                batch_y_test[example] = y_test
            else:
                # find configuration
                best_adaptor, best_feature = re.match(r"adaptor_(\d+)_feature_(\d+)", best_method).groups()
                
                ux_train, ux_test, ux_mean, ux_std, uy_train, uy_mean, uy_std = self.preprocess_data(
                    example=example,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    feature_names=self.validate_features[int(best_feature)]["features"]
                )

                adaptor_method = self.select_adaptor(self.validate_models[int(best_adaptor)]["model"])

                params=self.validate_models[int(best_adaptor)]
                if self.validate_models[int(best_adaptor)]["model"] == "nadaraya_watson":
                    params= {
                        "width": self.nadaraya_watson_opt_params[int(best_feature)]["width"][example],
                        "variance": self.nadaraya_watson_opt_params[int(best_feature)]["variance"][example],
                    }
                    
                y_test = adaptor_method(
                    x_train=ux_train,
                    y_train=uy_train,
                    x_test=ux_test,
                    y_mean=uy_mean,
                    y_std=uy_std,
                    **params,
                )
                batch_y_test[example] = y_test
        
        return batch_y_test

    def preprocess_data(
        self,
        example: int,
        x_train,
        x_test,
        y_train,
        feature_names: List[str],
    ):
        '''
            example wise addition of feature_names and normalization
        '''
        train_length = x_train.shape[1]
        features = [torch.cat([x_train[example], x_test[example]], dim=0)]
        ux_mean = [self.y_mean[example]]
        ux_std = [self.y_std[example]]
        if "covariates" in feature_names:
            features.append(self.get_features("covariates")[example])
            ux_mean.append(self.cov_mean[example])
            ux_std.append(self.cov_std[example])
        if "past_k" in feature_names:
            features.append(self.get_features("past_k")[example])
            ux_mean.extend([self.y_mean[example]] * self.num_past_k)
            ux_std.extend([self.y_std[example]] * self.num_past_k)
        if "positions" in feature_names:
            features.append(self.get_features("positions")[example])
        
        ux_ = torch.cat(features, dim=-1).unsqueeze(0)
        ux_train = ux_[:, :train_length]
        ux_test = ux_[:, train_length:]

        ux_mean = torch.cat(ux_mean, dim=-1)
        ux_std = torch.cat(ux_std, dim=-1)

        if "positions" in feature_names:
            ux_train[..., :-self.num_pos] = (ux_train[..., :-self.num_pos] - ux_mean) / ux_std
            ux_test[..., :-self.num_pos] = (ux_test[..., :-self.num_pos] - ux_mean) / ux_std
        else:
            ux_train = (ux_train - ux_mean) / ux_std
            ux_test = (ux_test - ux_mean) / ux_std
        
        uy_train = y_train[example:example+1]
    
        uy_train = (uy_train - self.y_mean[example]) / self.y_std[example]

        return ux_train, ux_test, ux_mean, ux_std, uy_train, self.y_mean[example], self.y_std[example]
    
    def normalize(
        self,
        x_train,
        x_test,
        y_train,
        features
    ):
        x_mean = [self.y_mean]
        x_std = [self.y_std]
        for feature in features:
            if "covariates" in feature:
                x_mean.append(self.cov_mean)
                x_std.append(self.cov_std)
            elif "past_k" in feature:
                x_mean.extend([self.y_mean] * self.num_past_k)
                x_std.extend([self.y_std] * self.num_past_k)
        x_mean = torch.cat(x_mean, dim=-1)
        x_std = torch.cat(x_std, dim=-1)
        
        if "positions" in features:
            x_train[..., :-self.num_pos] = (x_train[..., :-self.num_pos] - x_mean) / x_std
            x_test[..., :-self.num_pos] = (x_test[..., :-self.num_pos] - x_mean) / x_std
        else:
            x_train = (x_train - x_mean) / x_std
            x_test = (x_test - x_mean) / x_std

        y_mean = self.y_mean
        y_std = self.y_std

        y_train = (y_train - y_mean) / y_std

        return x_train, x_test, x_mean, x_std, y_train, y_mean, y_std
    
    def evaluate_adaptors(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_val: Float[torch.Tensor, "batch num_val_examples num_features"],
        y_val: Float[torch.Tensor, "batch num_val_examples patch_size"],
        eval_metrics: Dict[str, List[Float[torch.Tensor, "batch"]]],
        **kwargs
    ) -> None:
        '''
            Evaluate all combinations of adaptors X features
            on MAE 

            All input data is NOT normalized.
        '''
        for f, feature in self.validate_features.items():
            #### here we add all features 
            ux_train, ux_val = self.append_features(x_train, x_val, feature["features"])

            ux_train, ux_val, ux_mean, ux_std, uy_train, uy_mean, uy_std = self.normalize(ux_train, ux_val, y_train.clone().detach(), feature["features"])
            
            for a, adaptor in tqdm(self.validate_models.items(), desc="Validate models: ", leave=False):
                if adaptor["model"] == "gaussian_process":
                    optimized_params = self.optimize_gp_kernel_params(
                            x_train=ux_train.clone().detach(), 
                            y_train=uy_train.clone().detach(), 
                            x_test=ux_val.clone().detach(), 
                            y_true=y_val,
                            y_mean=uy_mean,
                            y_std=uy_std,
                            **adaptor ## params
                        )

                    for key, values in optimized_params.items():
                        if key not in self.gp_opt_params[a][f]:
                            self.gp_opt_params[a][f][key] = [values]
                        else:
                            self.gp_opt_params[a][f][key].append(values)

                    y_val_pred = self.eval_gaussian_process_estimator(
                            x_train=ux_train.clone().detach(), 
                            y_train=uy_train.clone().detach(), 
                            x_test=ux_val.clone().detach(), 
                            y_sub=y_val,
                            y_mean=uy_mean, 
                            y_std=uy_std, 
                            **{**adaptor, **optimized_params} # additional params
                        )
                    
                    for i, threshold in enumerate([None]+adaptor["threshold"]):
                        if f"adaptor_{a}_feature_{f}_threshold_{i}" not in eval_metrics:
                            eval_metrics[f"adaptor_{a}_feature_{f}_threshold_{i}"] = []

                        error = self.validation_metric(y_val, y_val_pred[i])
                        eval_metrics[f"adaptor_{a}_feature_{f}_threshold_{i}"].append(error)
                else:
                    if f"adaptor_{a}_feature_{f}" not in eval_metrics:
                        eval_metrics[f"adaptor_{a}_feature_{f}"] = []

                    if adaptor["model"] == "nadaraya_watson":
                        # find optimal width and variance
                        opt_width, opt_variance = self.optimize_nw_kernel_params(
                            x_train=ux_train.clone().detach(), 
                            y_train=uy_train.clone().detach(), 
                            x_test=ux_val.clone().detach(), 
                            y_true=y_val,
                            y_mean=uy_mean,
                            y_std=uy_std,
                        )

                        if "width" not in self.nadaraya_watson_opt_params[f] and "variance" not in self.nadaraya_watson_opt_params[f]:
                            self.nadaraya_watson_opt_params[f] = {"width": [opt_width], "variance": [opt_variance]}
                        else:
                            self.nadaraya_watson_opt_params[f]["width"].append(opt_width)
                            self.nadaraya_watson_opt_params[f]["variance"].append(opt_variance)

                        y_val_pred = self.eval_nadaraya_watson_kernel_estimator(
                                x_train=ux_train.clone().detach(), 
                                y_train=uy_train.clone().detach(), 
                                x_test=ux_val.clone().detach(), 
                                y_mean=uy_mean,
                                y_std=uy_std, 
                                width=opt_width,
                                variance=opt_variance
                        )
                    else:
                        adaptor_method = self.select_adaptor(adaptor["model"])
                        y_val_pred = adaptor_method(
                                x_train=ux_train.clone().detach(), 
                                y_train=uy_train.clone().detach(), 
                                x_test=ux_val.clone().detach(), 
                                y_mean=uy_mean,
                                y_std=uy_std, 
                                **adaptor
                            )
                        
                    error = self.validation_metric(y_val, y_val_pred)
                    eval_metrics[f"adaptor_{a}_feature_{f}"].append(error)

    def select_adaptor(self, model: str) -> Any:
        if model == "kernel_regression":
            return self.eval_kernel_regression_estimator
        elif model == "nadaraya_watson":
            return self.eval_nadaraya_watson_kernel_estimator
        elif model == "gaussian_process":
            return self.eval_gaussian_process_estimator
        elif model == "gradient_boosting":
            return self.eval_gradient_boosting_estimator
        elif model == "bayesian":
            return self.eval_bayesian_ridge_estimator
        else:
            raise ValueError(f"Invalid adaptor: {model}")
        
    def optimize_nw_kernel_params(
        self,
        x_train,
        y_train,
        x_test,
        y_true,
        y_mean,
        y_std,
    ):
        def objective(trial, i, x_train, y_train, x_test, y_true, y_mean, y_std):
            # width = trial.suggest_loguniform('width', 0.001, 10)
            # variance = trial.suggest_loguniform('variance', 0.1, 10)
            width = trial.suggest_loguniform('width', 0.01, 1)
            variance = trial.suggest_loguniform('variance', 0.1, 4)
            y_pred = self.eval_nadaraya_watson_kernel_estimator(
                    x_train=x_train[i:i+1], 
                    y_train=y_train[i:i+1],
                    x_test=x_test[i:i+1],
                    y_mean=y_mean[i:i+1],
                    y_std=y_std[i:i+1],
                    width=width,
                    variance=variance,
                )

            error = self.validation_metric(y_true[i:i+1], y_pred).mean()
            return error.item()
    
        def optimize_per_sample(i):
            study = optuna.create_study(sampler=TPESampler())
            study.optimize(lambda trial: objective(trial, i, x_train, y_train, x_test, y_true, y_mean, y_std), n_trials=32)
            return study.best_params['width'], study.best_params['variance']
            
        results = Parallel(n_jobs=16)(
            delayed(optimize_per_sample)(i)
            for i in tqdm(range(x_train.shape[0]), desc="Optimize NW") 
        )

        opt_widths = [res[0] for res in results]
        opt_variances = [res[1] for res in results]

        opt_widths = torch.tensor(opt_widths, device=x_train.device)
        opt_variances = torch.tensor(opt_variances, device=x_train.device)

        return opt_widths, opt_variances

    def optimize_gp_kernel_params(
        self,
        x_train,
        y_train,
        x_test,
        y_true,
        y_mean,
        y_std,
        **kwargs,
    ):
        kernel_type = kwargs.get('kernel_type', '')
        assert kernel_type != ""
        kernel_params = {
            "rbf": lambda trial: {
                "width": trial.suggest_loguniform("width", 0.1, 2),
                "variance": trial.suggest_loguniform("variance", 0.1, 4),
            },
            "matern": lambda trial: {
                "width": trial.suggest_loguniform("width", 0.1, 2),
                "variance": trial.suggest_loguniform("variance", 0.1, 4),
                "nu": trial.suggest_categorical("nu", [1.5, 2.5]),
            },
            "periodic": lambda trial: {
                "width": trial.suggest_loguniform("width", 0.5, 2),  # Avoid extreme small values
                "variance": trial.suggest_loguniform("variance", 0.1, 2),  # Reduce max variance
                "period": trial.suggest_uniform("period", 2, 6),  # Constrain period range
            },
        }
        def objective(trial, i, x_train, y_train, x_test, y_true, y_mean, y_std):
            hyperparams = kernel_params[kernel_type](trial)
            local_kwargs = {**kwargs, **hyperparams, "threshold": []}


            y_pred = self.eval_gaussian_process_estimator(
                    x_train=x_train[i:i+1], 
                    y_train=y_train[i:i+1],
                    x_test=x_test[i:i+1],
                    y_sub=y_true[i:i+1],
                    y_mean=y_mean[i:i+1],
                    y_std=y_std[i:i+1],
                    **local_kwargs
                )[0]

            error = self.validation_metric(y_true[i:i+1], y_pred).mean()
            return error.item()
    
        def optimize_per_sample(i):
            study = optuna.create_study(sampler=TPESampler())
            study.optimize(lambda trial: objective(trial, i, x_train, y_train, x_test, y_true, y_mean, y_std), n_trials=32)
            return study.best_params
            
        results = Parallel(n_jobs=16)(
            delayed(optimize_per_sample)(i)
            for i in tqdm(range(x_train.shape[0]), desc=f"Optimize GP kernel={kernel_type}") 
        )

        optimized_params = {key: [] for key in results[0]}

        for res in results:
            for key in optimized_params:
                optimized_params[key].append(res[key])

        for key in optimized_params:
            optimized_params[key] = torch.tensor(optimized_params[key], device=x_train.device)

        return optimized_params

    def get_features(
        self,
        feature: str,
        examples: Optional[Float[torch.Tensor, "examples"]] = None,
    ) -> Float[torch.Tensor, "examples num_toal_examples num_features"]:
        if feature in self.features:
            if examples is None:
                return self.features[feature]
            else:
                return self.features[feature][examples]
        else:
            raise ValueError(f"`{feature}` not defined.")

    def append_features(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        features: List[str],
        examples: Optional[Float[torch.Tensor, "examples"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch num_train_examples num_features_new"],
        Float[torch.Tensor, "batch num_test_examples num_features_new"]
    ]:
        '''
            x_train: time series y_i
            x_test: time series y_i

            features: 
                position: 
                past_k:
                covariates: 
        '''
        for feature in features:
            if feature in self.features:
                x_train, x_test = self.concat_feature(x_train, x_test, feature, examples)
            else:
                raise ValueError(f"{feature} is not available.")
        return x_train, x_test

    def eval_kernel_regression_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]],
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]],
        **kwargs
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        '''
        assert y_train.shape[0] == y_mean.shape[0]

        kernel = kwargs.get('kernel', "rbf")
        alpha = kwargs.get('alpha', 1.0)
        y_test = self.kernel_regression_estimator(x_train, y_train, x_test, kernel, alpha)
        
        y_test = y_test * y_std + y_mean
        return y_test
    
    def eval_bayesian_ridge_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]],
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]],
        **kwargs
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        '''
        assert y_train.shape[0] == y_mean.shape[0]

        y_test = self.bayesian_ridge_estimator(x_train, y_train, x_test)
        
        y_test = y_test * y_std + y_mean
        return y_test
    
    def eval_gradient_boosting_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]],
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]],
        **kwargs
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        '''
        assert y_train.shape[0] == y_mean.shape[0]

        params = kwargs.get("params", {"loss": "squared_error"})
        y_test = self.gradient_boosting_estimator(x_train, y_train, x_test, params)
        
        y_test = y_test * y_std + y_mean
        return y_test
    
    def eval_gradient_boosting_only_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]],
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]],
        **kwargs
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        '''
        assert y_train.shape[0] == y_mean.shape[0]

        params = kwargs.get("params", {"loss": "squared_error"})
        y_test = self.gradient_boosting_only_estimator(x_train, y_train, x_test, params)
        
        y_test = y_test * y_std + y_mean
        return y_test
        
    def eval_nadaraya_watson_kernel_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]] = None,
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]] = None,
        **kwargs
    ) -> Float[torch.Tensor, "batch num_train_examples patch_size"]:
        '''
        '''
        assert y_train.shape[0] == y_mean.shape[0]
        
        width = kwargs.get('width', 0.85)
        variance = kwargs.get('variance', 1.0)

        y_test = self.nadaraya_watson_kernel_estimator(x_train, y_train, x_test, width, variance)
        
        y_test = y_test * y_std + y_mean
        return y_test
        
    def eval_gaussian_process_estimator(
        self, 
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        y_sub: Float[torch.Tensor, "batch num_test_examples patch_size"],
        y_mean: Optional[Float[torch.Tensor, "batch 1 1"]],
        y_std: Optional[Float[torch.Tensor, "batch 1 1"]],
        **kwargs
    ) -> List[
        Float[torch.Tensor, "batch num_test_examples patch_size"]
    ]:
        '''
            GP estimator evaluating 
                without filtering using variance
                with filtering using variance

            validate best method
        '''
        assert y_train.shape[0] == y_mean.shape[0]
        
        variance_threshold = kwargs.get('threshold', [])

        y_test, covariance = self.gaussian_process_estimator(x_train, y_train, x_test, **kwargs)

        variance = torch.diagonal(covariance, dim1=1, dim2=2).unsqueeze(-1) 

        y_test = y_test * y_std + y_mean

        variance_filtered_y_test = [y_test]
        for threshold in variance_threshold:
            y_test_filtered = torch.where(variance > threshold, y_sub, y_test)
            variance_filtered_y_test.append(y_test_filtered)
        
        return variance_filtered_y_test
    
    def concat_feature(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        feature: str,
        examples: Optional[Float[torch.Tensor, "examples"]] = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch num_train_examples num_features_new"],
        Float[torch.Tensor, "batch num_test_examples num_features_new"],
    ]:
        '''
            x_train [--train--|--val--] <-- for cross validation
            x_test  [--val/test--]
            Feature [--train--|--val--|--test--]

        '''
        num_train_examples = x_train.shape[1]
        num_test_examples = x_test.shape[1]

        if examples is None:
            feature_train = self.features[feature][:, :num_train_examples]
            feature_test = self.features[feature][:, num_train_examples: (num_train_examples+num_test_examples)]
        else:
            feature_train = self.features[feature][examples, :num_train_examples]
            feature_test = self.features[feature][examples, num_train_examples: (num_train_examples+num_test_examples)]

        x_train = torch.cat([x_train, feature_train], dim=-1)
        x_test = torch.cat([x_test, feature_test], dim=-1)

        return x_train, x_test
    
    def rbf_kernel(
        self,
        x_1: Float[torch.Tensor, "batch num_examples num_features"],
        x_2: Float[torch.Tensor, "batch num_examples num_features"],
        width: float,
        variance: float,
        **kwargs
    ) -> Float[torch.Tensor, "batch num_examples num_examples"]:
        '''
            compute batch rbf kernel

            variance * exp( - || x_1 - x_2 || ^ 2 / 2 * width ^ 2)

        '''
        if type(width) == torch.Tensor:
            width = width.reshape(-1, 1, 1)
            variance = variance.reshape(-1, 1, 1)
        squared_dist = torch.cdist(x_1, x_2, p=2, compute_mode='use_mm_for_euclid_dist') ** 2
        return variance * torch.exp(- squared_dist / (2 * (width ** 2)))
    
    def matern_kernel(
        self,
        x_1: Float[torch.Tensor, "batch num_examples num_features"],
        x_2: Float[torch.Tensor, "batch num_examples num_features"],
        width: float,
        variance: float,
        nu: float,
        **kwargs
    ) -> Float[torch.Tensor, "batch num_examples num_examples"]:
        '''

        '''
        if type(width) == torch.Tensor:
            width = width.reshape(-1, 1, 1)
            variance = variance.reshape(-1, 1, 1)
        
        dist = torch.cdist(x_1, x_2, p=2, compute_mode='use_mm_for_euclid_dist')

        if not isinstance(nu, torch.Tensor):
            nu = torch.tensor([nu], device=dist.device, dtype=dist.dtype)
        if nu.shape == ():
            nu = nu.expand(dist.shape[0])  # Expand scalar to batch size

        sqrt_3_d = (3 ** 0.5) * dist / width
        sqrt_5_d = (5 ** 0.5) * dist / width
        # K_05 = torch.exp(-dist / width)
        K_15 = (1 + sqrt_3_d) * torch.exp(-sqrt_3_d)
        K_25 = (1 + sqrt_5_d + (5/3) * (dist**2 / width**2)) * torch.exp(-sqrt_5_d)

        # K = torch.where(nu.reshape(-1, 1, 1) == 0.5, K_05, 
        #     torch.where(nu.reshape(-1, 1, 1) == 1.5, K_15, K_25)
        # )

        K = torch.where(nu.reshape(-1, 1, 1) == 1.5, K_15, K_25)

        return variance * K
    
    def periodic_kernel(
        self,
        x_1: Float[torch.Tensor, "batch num_examples num_features"],
        x_2: Float[torch.Tensor, "batch num_examples num_features"],
        period: float,
        width: float,
        variance: float,
        **kwargs
    ) -> Float[torch.Tensor, "batch num_examples num_examples"]:
        '''
        '''
        if type(width) == torch.Tensor:
            width = width.reshape(-1, 1, 1)
            variance = variance.reshape(-1, 1, 1)
            period = period.reshape(-1, 1, 1)
        dist = torch.cdist(x_1, x_2, p=2, compute_mode='use_mm_for_euclid_dist')
        sin_x = torch.sin((torch.pi * dist) / period) ** 2
        return variance * torch.exp(- 2 * sin_x / (width ** 2))
    
    def hybrid_kernel(
        self,
        yhat_1, 
        yhat_2,
        cov_1, 
        cov_2,
        pastk_1,
        pastk_2,
        pos_1,
        pos_2,
        matern_width,
        matern_variance,
        nu,
        rbf_width,
        rbf_variance,
        period,
        period_width,
        period_variance
    ):
        '''
            covariates 
            yhat predicted values
            pos position
            pastk
        '''
        k_matern = self.matern_kernel(yhat_1, yhat_2, matern_width, matern_variance, nu=nu)
        k_rbf = self.rbf_kernel(cov_1, cov_2, rbf_width, rbf_variance)
        k_linear = torch.bmm(yhat_1, yhat_2.transpose(1,2))
        k_periodic = self.periodic_kernel(pos_1, pos_2, period, period_width, period_variance)

        return k_matern + k_rbf + k_linear + k_periodic

    def kernel_regression_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        kernel: str,
        alpha: float
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch kernel regression
        '''
        bsz, _, target_dim = y_train.shape
        batch_y_test = torch.empty(bsz, self.prediction_length, target_dim, device=y_train.device)

        for i in tqdm(range(bsz), desc="Kernel Regression Estimator: ", leave=False):
            regr_model = KernelRidge(kernel=kernel, alpha=alpha)
            regr_model.fit(x_train[i], y_train[i])
            try:
                y_test = regr_model.predict(x_test[i]).get()
                batch_y_test[i] = torch.from_numpy(y_test).to(device=y_train.device)
            except Exception as e:
                print(f"Exception occured: {e}.\nOverriding by setting forecast to very large values.") 
                y_test = torch.tensor([torch.finfo(torch.float32).max], device=batch_y_test.device).unsqueeze(-1).repeat(self.prediction_length, target_dim)
                batch_y_test[i] = y_test

        return batch_y_test

    def bayesian_ridge_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch kernel regression
        '''
        bsz, _, target_dim = y_train.shape
        batch_y_test = torch.empty(bsz, self.prediction_length, target_dim, device=y_train.device)

        for i in tqdm(range(bsz), desc="Bayesian Regression Estimator: ", leave=False):
            regr_model = BayesianRidge()
            regr_model.fit(x_train[i].cpu().numpy(), y_train[i].squeeze().cpu().numpy())
            try:
                y_test = regr_model.predict(x_test[i].cpu().numpy())
                batch_y_test[i] = torch.from_numpy(y_test).to(device=y_train.device).unsqueeze(-1)
            except Exception as e:
                print(f"Exception occured: {e}.\nOverriding by setting forecast to very large values.") 
                y_test = torch.tensor([torch.finfo(torch.float32).max], device=batch_y_test.device).unsqueeze(-1).repeat(self.prediction_length, target_dim)
                batch_y_test[i] = y_test

        return batch_y_test
    
    def gradient_boosting_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        params: Dict[str, Any]
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch kernel regression
        '''
        bsz, _, target_dim = y_train.shape
        batch_y_test = torch.empty(bsz, self.prediction_length, target_dim, device=y_train.device)

        def train_and_predict(i, x_train, y_train, x_test):
            try:
                regr_model = XGBRegressor(tree_method="hist", device="cuda") #GradientBoostingRegressor(**params)
                regr_model.fit(x_train[i], y_train[i].squeeze())
                y_test = regr_model.predict(x_test[i])
                return i, torch.from_numpy(y_test).unsqueeze(-1).to(device=y_train.device)
            except Exception as e:
                print(f"Exception occured in GB only method: {e}.\nOverriding by setting forecast to very large values.") 
                y_test = torch.tensor([torch.finfo(torch.float32).max], device=batch_y_test.device).unsqueeze(-1).repeat(self.prediction_length, target_dim)
                return i, y_test
            
        results = Parallel(n_jobs=16)(delayed(train_and_predict)(i, x_train, y_train, x_test) for i in tqdm(range(bsz)))

        for i, y_test in results:
            batch_y_test[i] = y_test

        return batch_y_test
    
    def gradient_boosting_only_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        params: Dict[str, Any]
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch kernel regression
        '''
        bsz, _, target_dim = y_train.shape
        batch_y_test = torch.empty(bsz, self.prediction_length, target_dim, device=y_train.device)

        def train_and_predict(i, x_train, y_train, x_test):
            try:
                regr_model = XGBRegressor(tree_method="hist", device="cuda") #GradientBoostingRegressor(**params)
                regr_model.fit(x_train[i], y_train[i].squeeze())
                y_test = regr_model.predict(x_test[i])
                return i, torch.from_numpy(y_test).unsqueeze(-1).to(device=y_train.device)
            except Exception as e:
                print(f"Exception occured in GB only method: {e}.\nOverriding by setting forecast to very large values.") 
                y_test = torch.tensor([torch.finfo(torch.float32).max], device=batch_y_test.device).unsqueeze(-1).repeat(self.prediction_length, target_dim)
                return i, y_test
            
        results = Parallel(n_jobs=16)(delayed(train_and_predict)(i, x_train, y_train, x_test) for i in tqdm(range(bsz)))

        for i, y_test in results:
            batch_y_test[i] = y_test

        return batch_y_test
    
    def nadaraya_watson_kernel_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        width: float,
        variance: float,
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch nadaraya watson kernel estimator

            y = sum_{i=1}^{i=N} K(x, x_i) * y_i / sum_{i=1}^{i=N} K(x, x_i)
        '''
        K = self.rbf_kernel(x_train, x_test, width, variance)
        y_test = torch.bmm(K.transpose(1,2), y_train) / (torch.sum(K, dim=1).unsqueeze(-1) + self.eps)

        if torch.isnan(y_test).any():
            raise ValueError(f"NW estimator resulting in NaN.")

        return y_test

    def gaussian_process_estimator(
        self,
        x_train: Float[torch.Tensor, "batch num_train_examples num_features"],
        y_train: Float[torch.Tensor, "batch num_train_examples patch_size"],
        x_test: Float[torch.Tensor, "batch num_test_examples num_features"],
        kernel_type: str,
        **kwargs
    ) -> Tuple[
        Float[torch.Tensor, "batch num_test_examples patch_size"],
        Float[torch.Tensor, "batch num_test_exampels num_test_examples"]
    ]:
        '''

            batch gaussian process estimator

            mean = mu_y + K_s.T * K_inv * y 
            cov = K_ss - K_s.T * K_inv * K_s
        '''
        if kernel_type == "rbf":
            K = self.rbf_kernel(x_train, x_train, **kwargs)
            K_s = self.rbf_kernel(x_train, x_test, **kwargs)
            K_ss = self.rbf_kernel(x_test, x_test, **kwargs)
        elif kernel_type == "matern":
            K = self.matern_kernel(x_train, x_train, **kwargs)
            K_s = self.matern_kernel(x_train, x_test, **kwargs)
            K_ss = self.matern_kernel(x_test, x_test, **kwargs)
        elif kernel_type == "periodic":
            K = self.periodic_kernel(x_train, x_train, **kwargs)
            K_s = self.periodic_kernel(x_train, x_test, **kwargs)
            K_ss = self.periodic_kernel(x_test, x_test, **kwargs)

        K_reg = K + torch.eye(K.shape[1], device=K.device).unsqueeze(0) + self.eps

        K_inv, info = torch.linalg.cholesky_ex(K_reg)

        if torch.any(info != 0):  # If not positive definite
            print("Warning: K_reg is not positive definite! Increasing jitter...")
            jitter = 1e-4
            K_reg += jitter * torch.eye(K.shape[1], device=K.device).unsqueeze(0)
            L = torch.linalg.cholesky(K_reg)

        alpha = torch.cholesky_solve(y_train, K_inv)

        y_mean = torch.bmm(K_s.transpose(1,2), alpha)

        v = torch.cholesky_solve(K_s, K_inv)

        y_cov = K_ss - torch.bmm(K_s.transpose(1,2), v)

        return y_mean, y_cov
