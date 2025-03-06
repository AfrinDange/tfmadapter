import csv
import cupy
import logging
import os
import re
import torch
import warnings


import optuna
from optuna.samplers import TPESampler

from collections import defaultdict
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class DataClassConfig:
    name: str
    context_length: int
    smaller_context: int
    smallest_history: int
    rolling_window_size: int 
    prediction_length: int
    seasonality: int
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

class TwoStageGPAdaptor:
    '''
    
    '''
    def __init__(
        self,
        ts_pred_history: Float[torch.Tensor, "batch num_train_examples patch_size"],
        ts_pred_future: Float[torch.Tensor, "batch num_test_examples num_features"],
        ts_true_history: Float[torch.Tensor, "batch num_train_examples patch_size"],
        y_mean: Float[torch.Tensor, "batch 1 1"],
        y_std: Float[torch.Tensor, "batch 1 1"],
        cov_mean: Float[torch.Tensor, "batch 1 num_covariates"],
        cov_std: Float[torch.Tensor, "batch 1 num_covariates"],
        stage_one_features: Dict[str, Float[torch.Tensor, "batch num_total_examples feature_dim"]],
        history_features: Dict[str, Float[torch.Tensor, "batch num_total_examples feature_dim"]],
        future_features: Dict[str, Float[torch.Tensor, "batch num_total_examples feature_dim"]],
        ds_config: Dict[str, Any],
        ts_true_future: Optional[Float[torch.Tensor, "batch pred_length patch_size"]] = None,
    ):
        '''
        
        '''
        self.ds_config = DataClassConfig(**ds_config)
        self.entire_context_length = stage_one_features["all_history"].shape[1]
        self.prediction_length = self.ds_config.prediction_length
        self.num_covariates = stage_one_features["covariates"].shape[-1]
        self.num_past_k = stage_one_features["past_k"].shape[-1]
        self.num_pos = stage_one_features["all_positions"].shape[-1]
        self.validation_metric = mse if self.ds_config.validation_metric == "mse" else mae
        self.has_cat_covariates = history_features["cat_covariates"] is not None

        self.history_features = history_features
        self.future_features = future_features

        # normalization params
        self.eps = 1e-6
        self.y_mean = y_mean
        self.y_std = y_std + self.eps
        self.cov_mean = cov_mean
        self.cov_std = cov_std + self.eps 

        self.fm_predictions_train = ts_pred_history
        self.fm_predictions_test = ts_pred_future
        self.y_true = ts_true_history

        self.future_target = ts_true_future

        # features and config
        self.features = stage_one_features

        stage_one_features["all_past_k"] = torch.cat([
                stage_one_features["all_history"][:,:1].repeat(1, self.num_past_k, 1),
                stage_one_features["all_history"],
                self.fm_predictions_test[:, :-1]
            ], dim=1).unfold(dimension=1, size=self.num_past_k, step=1).squeeze(-2)
        
        assert stage_one_features["all_past_k"].shape[1] == self.entire_context_length + self.prediction_length, f"{stage_one_features['all_past_k'].shape} current shape"

        # positions 
        stage_one_features["all_positions"] = stage_one_features["all_positions"].unsqueeze(0).repeat(self.y_true.shape[0], 1, 1)

        self.validate_features = {id:{"features": features} for id, features in enumerate(self.ds_config.features_for_selection)}

        self.gp_opt_params = defaultdict(lambda: defaultdict(lambda: defaultdict(list))) 
        # self.optimized_weights = defaultdict(list)


        ################# 

        # normalize position ids

        self.history_features["position_ids"] = self.history_features["position_ids"] / self.ds_config.seasonality
        self.future_features["position_ids"] = self.future_features["position_ids"] / self.ds_config.seasonality


        # self.history_features["covariates"] = torch.cat([
        #     self.history_features["covariates"][:, :1].repeat(1, self.num_past_k, 1),
        #     self.history_features["covariates"]
        # ], dim=1).unfold(dimension=1, step=1, size=self.num_past_k+1).flatten(start_dim=2)

        # self.future_features["covariates"] = torch.cat([
        #     self.future_features["covariates"][:, :1].repeat(1, self.num_past_k, 1),
        #     self.future_features["covariates"]
        # ], dim=1).unfold(dimension=1, step=1, size=self.num_past_k+1).flatten(start_dim=2)


        for feature in self.history_features:
            print(feature, self.history_features[feature].shape if self.history_features[feature] is not None else None)
            print(feature, self.future_features[feature].shape if self.history_features[feature] is not None else None)
            
    def generate_training_data(
        self,
        regressor="bayesian"
    ):
        '''
        
        '''
        batch_size, train_length, dim = self.y_true.shape

        x_train = self.y_true 
        x_test = self.get_features("all_history")

        x_train = torch.cat([
            x_train,
            self.get_features("past_k")[:, :train_length],
            self.get_features("positions")[:, :train_length]
        ], dim=-1)

        x_test = torch.cat([
            x_test,
            self.get_features("all_past_k")[:, :self.entire_context_length], # past_k
            self.get_features("all_positions")[:, :self.entire_context_length]
        ], dim=-1) 

        x_test__ = torch.cat([
            self.fm_predictions_test,
            self.get_features("all_past_k")[:, self.entire_context_length:], # past_k
            self.get_features("all_positions")[:, self.entire_context_length:]
        ], dim=-1) 

        # output feature --- FM prediction
        y_train = self.fm_predictions_train

        # normalize
        # x_mean = x_train[..., :-self.num_pos].mean(dim=1, keepdim=True)
        # x_std = x_train[..., :-self.num_pos].std(dim=1, keepdim=True) + self.eps

        x_mean = torch.cat([
            self.y_mean,
            *([self.y_mean] * self.num_past_k)
        ], dim=-1)
        x_std = torch.cat([
            self.y_std,
            *([self.y_std] * self.num_past_k)
        ], dim=-1)

        x_train[..., :-self.num_pos] = (x_train[..., :-self.num_pos] - x_mean) / x_std
        x_test[..., :-self.num_pos] = (x_test[..., :-self.num_pos] - x_mean) / x_std
        x_test__[..., :-self.num_pos] = (x_test__[..., :-self.num_pos] - x_mean) / x_std

        y_mean = self.y_mean
        y_std = self.y_std

        y_train = (y_train - y_mean) / y_std

        # fit a model, predict FM forecasts for all points in history
        data = torch.empty(batch_size, self.entire_context_length, dim, device=x_train.device)
        data__ = torch.empty(batch_size, self.prediction_length, dim, device=x_test__.device)
        if regressor == "bayesian":
            for i in tqdm(range(batch_size), desc="generate training data: ", leave=False):
                data_model = BayesianRidge()
                data_model.fit(x_train[i].cpu().numpy(), y_train[i].squeeze(-1).cpu().numpy())
                data[i] = torch.from_numpy(data_model.predict(x_test[i].cpu().numpy())).to(x_test.device).unsqueeze(-1)

                data__[i] = torch.from_numpy(data_model.predict(x_test__[i].cpu().numpy())).to(x_test.device).unsqueeze(-1)
        else:
            raise ValueError(f"Pseudo-forecast generator {regressor} not implemented")
        
        data = data * y_std + y_mean
        data__ = data__ * y_std + y_mean
        
        # create train, test data
        # x_train --- fm forecasts (data)
        # y_train --- true values (all_history)
        # x_test --- real fm forecasts from actual horizon

        self.x_train = data
        self.y_train = self.get_features("all_history")
        self.x_test =  data__ 
        # self.x_test = self.fm_predictions_test

        assert self.x_train.shape == (batch_size, self.entire_context_length, dim), f"x_train shape is {self.x_train.shape}, expected {(batch_size, self.entire_context_length, dim)}"
        assert self.y_train.shape == (batch_size, self.entire_context_length, dim), f"y_train shape is {self.y_train.shape}, expected {(batch_size, self.entire_context_length, dim)}"

        return self.x_train, self.y_train, self.x_test

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

    def _cross_validate(
        self
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        
        '''
        num_examples = self.ts_pred_history.shape[1]

        # create cross validation sets
        folds=self.ds_config.folds
        error_per_fold = []
        for i in tqdm(range(folds, 0, -1), desc=f"{folds}-fold validation", leave=False):
            train_length = num_examples - (i * self.prediction_length)
            
            data_inputs = self.preprocess_data_validation(train_length)
            errors = self.evaluate_adaptor(data_inputs)
            error_per_fold.append(errors)

        error = torch.cat([torch.cat([err.unsqueeze(0) for err in error], dim=0).unsqueeze(0) for error in error_per_fold], dim=0)
        print(error.shape)
        self.threshold = error.mean(dim=0).argmin(dim=0)
        
        print(self.threshold.shape, self.threshold)

        self.gp_opt_params = {
            feature: {
                kernel: {param: torch.stack(val).mean(dim=0) for param, val in params.items()} for kernel, params in kernels.items()
            } for feature, kernels in self.gp_opt_params.items()
        }

        # self.optimized_weights = {
        #     key: torch.stack(val).mean(dim=0) for key, val in self.optimized_weights.items()
        # }

    def predict(
        self,
    ):
        batch_size = self.ts_pred_history.shape[0]

        batch_ts_test = torch.empty((batch_size, self.prediction_length, 1), device=self.ts_pred_history.device)

        data_inputs = self.preprocess_data_test()

        for example in tqdm(range(batch_size), desc="Final Predictions: ", leave=False):
            data_input = {key: val[example:example+1] if val is not None else None for key, val in data_inputs.items()}

            kwargs = {
                feature: {
                    kernel: {param: val[example] for param, val in params.items()} for kernel, params in kernels.items()
                } for feature, kernels in self.gp_opt_params.items()
            }

            # weights = {
            #     key: val[example] for key, val in self.optimized_weights.items()
            # }

            threshold = self.threshold[example]

            # ts_test = self.eval_gaussian_process_estimator(
            #     **data_input,
            #     **{**kwargs, **weights, "threshold": [] if threshold == 0 else [[2.0, 2.5, 3.0][threshold-1]]}
            # )[-1]

            ts_test = self.eval_gaussian_process_estimator(
                **data_input,
                **{**kwargs, "threshold": [] if threshold == 0 else [[2.0, 2.5, 3.0][threshold-1]]}
            )[-1]

            batch_ts_test[example] = ts_test

        return batch_ts_test
    
    def predict_with_gb(
        self,
    ):
        batch_size = self.ts_pred_history.shape[0]

        data_inputs = self.preprocess_data_test()
        
        ts_test = self.eval_gradient_boosting_estimator(
            **data_inputs,
        )
        return ts_test

    def preprocess_data_test(
        self
    ):
        '''
        given train length, create train|val data
        '''
        data_inputs = dict()

        data_inputs["ts_pred_history"] = self.ts_pred_history
        data_inputs["ts_true_history"] = self.ts_true_history
        data_inputs["ts_past_k_history"] = self.history_features["past_k"]
        data_inputs["covariates_history"] = self.history_features["covariates"]
        data_inputs["cat_covariates_history"] = self.history_features["cat_covariates"] if self.has_cat_covariates else None
        data_inputs["position_ids_history"] = self.history_features["position_ids"]

        # create validation data from history
        data_inputs["ts_pred_future"] = self.ts_pred_future
        data_inputs["ts_past_k_future"] = self.future_features["past_k"]
        data_inputs["covariates_future"] = self.future_features["covariates"]
        data_inputs["cat_covariates_future"] = self.future_features["cat_covariates"] if self.has_cat_covariates else None
        data_inputs["position_ids_future"] = self.future_features["position_ids"]

        ts_mean = torch.mean(data_inputs["ts_true_history"], dim=1, keepdim=True)            
        ts_std = torch.std(data_inputs["ts_true_history"], dim=1, keepdim=True) + self.eps

        cov_mean = torch.mean(data_inputs["covariates_history"], dim=1, keepdim=True)         
        cov_std = torch.std(data_inputs["covariates_history"], dim=1, keepdim=True) + self.eps

        # normalize
        data_inputs["ts_pred_history"] = (data_inputs["ts_pred_history"] - ts_mean) / ts_std
        data_inputs["ts_true_history"] = (data_inputs["ts_true_history"] - ts_mean) / ts_std
        data_inputs["ts_past_k_history"] = (data_inputs["ts_past_k_history"] - ts_mean) / ts_std
        data_inputs["covariates_history"] = (data_inputs["covariates_history"] - cov_mean) / cov_std
        
        data_inputs["ts_pred_future"] = (data_inputs["ts_pred_future"] - ts_mean) / ts_std
        data_inputs["ts_past_k_future"] = (data_inputs["ts_past_k_future"] - ts_mean) / ts_std
        data_inputs["covariates_future"] = (data_inputs["covariates_future"] - cov_mean) / cov_std

        data_inputs["ts_mean"] = ts_mean
        data_inputs["ts_std"] = ts_std

        return data_inputs

    def preprocess_data_validation(
        self,
        train_length,
    ):
        '''
        given train length, create train|val data
        '''
        data_inputs = dict()

        data_inputs["ts_pred_history"] = self.ts_pred_history[:, :train_length]
        data_inputs["ts_true_history"] = self.ts_true_history[:, :train_length]
        data_inputs["ts_past_k_history"] = self.history_features["past_k"][:, :train_length]
        data_inputs["covariates_history"] = self.history_features["covariates"][:, :train_length]
        data_inputs["cat_covariates_history"] = self.history_features["cat_covariates"][:, :train_length] if self.has_cat_covariates else None
        data_inputs["position_ids_history"] = self.history_features["position_ids"][:, :train_length]

        # create validation data from history
        data_inputs["ts_pred_future"] = self.ts_pred_history[:, train_length:train_length+self.prediction_length]
        data_inputs["ts_true_future"] = self.ts_true_history[:, train_length:train_length+self.prediction_length]
        data_inputs["ts_past_k_future"] = self.history_features["past_k"][:, train_length:train_length+self.prediction_length]
        data_inputs["covariates_future"] = self.history_features["covariates"][:, train_length:train_length+self.prediction_length]
        data_inputs["cat_covariates_future"] = self.history_features["cat_covariates"][:, train_length:train_length+self.prediction_length] if self.has_cat_covariates else None
        data_inputs["position_ids_future"] = self.history_features["position_ids"][:, train_length:train_length+self.prediction_length]

        ts_mean = torch.mean(data_inputs["ts_true_history"], dim=1, keepdim=True)            
        ts_std = torch.std(data_inputs["ts_true_history"], dim=1, keepdim=True) + self.eps

        cov_mean = torch.mean(data_inputs["covariates_history"], dim=1, keepdim=True)         
        cov_std = torch.std(data_inputs["covariates_history"], dim=1, keepdim=True) + self.eps

        # normalize
        data_inputs["ts_pred_history"] = (data_inputs["ts_pred_history"] - ts_mean) / ts_std
        data_inputs["ts_true_history"] = (data_inputs["ts_true_history"] - ts_mean) / ts_std
        data_inputs["ts_past_k_history"] = (data_inputs["ts_past_k_history"] - ts_mean) / ts_std
        data_inputs["covariates_history"] = (data_inputs["covariates_history"] - cov_mean) / cov_std
        
        data_inputs["ts_pred_future"] = (data_inputs["ts_pred_future"] - ts_mean) / ts_std
        data_inputs["ts_past_k_future"] = (data_inputs["ts_past_k_future"] - ts_mean) / ts_std
        data_inputs["covariates_future"] = (data_inputs["covariates_future"] - cov_mean) / cov_std

        data_inputs["ts_mean"] = ts_mean
        data_inputs["ts_std"] = ts_std

        return data_inputs

    def evaluate_adaptor(
        self,
        data_inputs,
    ):
        '''
        
        '''
        # optimized_params, optimized_weights = self.optimize_hyperparams(
        #     **{key: val.clone() if val is not None else None for key, val in data_inputs.items()}
        # )
        optimized_params, feature_set = self.optimize_hyperparams(
            **{key: val.clone() if val is not None else None for key, val in data_inputs.items()}
        )

        for feature in optimized_params:
            for kernel in optimized_params[feature]:
                for param, value in optimized_params[feature][kernel].items():
                    self.gp_opt_params[feature][kernel][param].append(value)

        self.feature_set.append(feature_set)

        # for weight, val in optimized_weights.items():
        #     self.optimized_weights[weight].append(val)

        # ts_val = self.eval_gaussian_process_estimator(
        #     **{key: val.clone() if val is not None else None for key, val in data_inputs.items()},
        #     **{**optimized_params, **optimized_weights, "threshold": [2.0, 2.5, 3.0]}                   
        # )
        ts_val = self.eval_gaussian_process_estimator(
            **{key: val.clone() if val is not None else None for key, val in data_inputs.items()},
            **{**optimized_params, "threshold": [2.0, 2.5, 3.0]}                   
        )

        errors = []        
        for i, threshold in enumerate([None, 2.0, 2.5, 3.0]):
            error = self.validation_metric(data_inputs["ts_true_future"], ts_val[i])
            errors.append(error)

        return errors


    def optimize_hyperparams(
        self,
        **data_inputs
    ):
        '''
        '''
        kernel_params = {
            "ts_kwargs": {
                "rbf": lambda trial: {
                    "width": trial.suggest_float(name="ts_kwargs.rbf.width", low=0.1, high=2, log=True),
                    "variance": trial.suggest_float(name="ts_kwargs.rbf.variance", low=0.1, high=4, log=True),
                },
                "linear": lambda trial: {
                    "variance": trial.suggest_float(name="ts_kwargs.linear.variance", low=0.1, high=2, log=True),
                },
            }, 
            "covariate_kwargs": {
                "rbf": lambda trial: {
                    "width": trial.suggest_float(name="covariate_kwargs.rbf.width", low=0.1, high=2, log=True),
                    "variance": trial.suggest_float(name="covariate_kwargs.rbf.variance", low=0.1, high=4, log=True),
                },
                "linear": lambda trial: {
                    "variance": trial.suggest_float(name="covariate_kwargs.linear.variance", low=0.1, high=2, log=True),
                },
            },
            "position_kwargs": {
                "periodic": lambda trial: {
                    "period": trial.suggest_float(name="position_kwargs.periodic.period", low=2, high=6, log=True),
                    "width": trial.suggest_float(name="position_kwargs.periodic.width", low=0.1, high=2, log=True),
                    "variance": trial.suggest_float(name="position_kwargs.periodic.variance", low=0.1, high=2, log=True),
                }
            },
        }

        # kernel_weights = lambda trial: {
        #     "ts_weight": trial.suggest_float(name="ts_weight", low=1e-3, high=1, log=True),
        #     "covariate_weight": trial.suggest_float(name="covariate_weight", low=1e-3, high=1, log=True),
        #     "position_weight": trial.suggest_float(name="position_weight", low=1e-3, high=1, log=True),
        # }


        if self.has_cat_covariates:
            kernel_params["cat_covariate_kwargs"] = {
                "rbf": lambda trial: {
                    "width": trial.suggest_float(name="cat_covariate_kwargs.rbf.width", low=0.1, high=2, log=True),
                    "variance": trial.suggest_float(name="cat_covariate_kwargs.rbf.variance", low=0.1, high=4, log=True),
                }
            }
            # kernel_weights = lambda trial: {
            #     "ts_weight": trial.suggest_float(name="ts_weight", low=0, high=1, log=True),
            #     "covariate_weight": trial.suggest_float(name="covariate_weight", low=0, high=1, log=True),
            #     "cat_covariate_weight": trial.suggest_float(name="cat_covariate_weight", low=0, high=1, log=True),
            #     "position_weight": trial.suggest_float(name="position_weight", low=0, high=1, log=True),
            # }
        

        def objective(
            trial,
            i,
            data_inputs,
        ):
            '''
            '''
            hyperparams = {
                feature: {
                    kernel: sampler(trial) for kernel, sampler in kernels.items()
                } for feature, kernels in kernel_params.items()
            }

            feature_choice = trial.suggest_categorical("feature_set", [
                "past_k covariates",
                "past_k positions covariates",
                "positions covariates"
            ])

            # weights = kernel_weights(trial)

            # raw_weights = torch.tensor([weights["ts_weight"], weights["covariate_weight"], weights["position_weight"]], dtype=torch.float32)
            # ts_weight, covariate_weight, position_weight = torch.nn.functional.softmax(raw_weights, dim=0).tolist()

            # c0 = ts_weight + covariate_weight + position_weight - 1
            # trial.set_user_attr("constraint", (c0,))

            # weights = {
            #     "ts_weight": ts_weight,
            #     "covariate_weight": covariate_weight,
            #     "position_weight": position_weight,
            # }


            # kwargs = { **hyperparams, **weights, "threshold": []}
            kwargs = { **hyperparams, "feature_set": feature_choice, "threshold": []}

            data_input = {key: val[i:i+1] if val is not None else None for key, val in data_inputs.items()}

            y_pred = self.eval_gaussian_process_estimator(
                **data_input,
                **kwargs
            )[0]

            error = self.validation_metric(data_input["ts_true_future"], y_pred).mean()
            return error.item()
        
        def optimize_per_sample(i):
            study = optuna.create_study(sampler=TPESampler())
            study.optimize(lambda trial: objective(
                trial,
                i,
                data_inputs
            ), n_trials=512)
            
            return study.best_params
        
        results = Parallel(n_jobs=16)(
            delayed(optimize_per_sample)(i) for i in tqdm(range(data_inputs["ts_pred_history"].shape[0]), desc=f"Optimize GP Kernel")
        )

        # results=[]

        # for i in tqdm(range(data_inputs["ts_pred_history"].shape[0]), desc=f"Optimize GP Kernel"):
        #     result = optimize_per_sample(i)
        #     results.append(result)

        optimized_params = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # optimized_weights = defaultdict(list)
        
        feature_set = []

        for res in results:
            for key, val in res.items():
                # if "weight" in key:
                #     optimized_weights[key].append(val)
                if key == "feature_set":
                    feature_set.append(val)
                else:
                    feature, kernel, param = key.split(".")
                    optimized_params[feature][kernel][param].append(val)

        optimized_params = {
            feature: {
                kernel: {
                    param: torch.tensor(values, device=data_inputs["ts_pred_history"].device)
                    for param, values in params.items()
                } for kernel, params in kernels.items()
            } for feature, kernels in optimized_params.items()
        }

        # optimized_weights = {
        #     key: torch.tensor(val, device=data_inputs["ts_pred_history"].device) for key, val in optimized_weights.items()
        # }

        # return optimized_params, optimized_weights
        return optimized_params, feature_set
    
    def eval_gaussian_process_estimator(
        self,
        ts_pred_history,
        ts_true_history,
        ts_pred_future,
        ts_past_k_history,
        ts_past_k_future,
        covariates_history,
        covariates_future,
        cat_covariates_history,
        cat_covariates_future,
        position_ids_history,
        position_ids_future,
        ts_mean,
        ts_std,
        **kwargs
    ):
        assert ts_pred_history.shape[0] == ts_mean.shape[0]

        variance_threshold = kwargs.get("threshold", [])

        ts_test, covariance = self.gaussian_process_estimator(
            ts_pred_history=ts_pred_history,
            ts_true_history=ts_true_history,
            ts_pred_future=ts_pred_future,
            ts_past_k_history=ts_past_k_history,
            ts_past_k_future=ts_past_k_future,
            covariates_history=covariates_history,
            covariates_future=covariates_future,
            cat_covariates_history=cat_covariates_history,
            cat_covariates_future=cat_covariates_future,
            position_ids_history=position_ids_history,
            position_ids_future=position_ids_future,
            **kwargs
        )

        variance = torch.diagonal(covariance, dim1=1, dim2=2).unsqueeze(-1)

        ts_test = ts_test * ts_std + ts_mean

        variance_filtered_ts_test = [ts_test]

        for threshold in variance_threshold:
            ts_test_filtered = torch.where(variance > threshold, ts_pred_future, ts_test)
            variance_filtered_ts_test.append(ts_test_filtered)
        
        return variance_filtered_ts_test

    def eval_gradient_boosting_estimator(
        self,
        ts_pred_history,
        ts_true_history,
        ts_pred_future,
        ts_past_k_history,
        ts_past_k_future,
        covariates_history,
        covariates_future,
        cat_covariates_history,
        cat_covariates_future,
        position_ids_history,
        position_ids_future,
        ts_mean,
        ts_std,
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
        '''
        ts_test = self.gradient_boosting_estimator(
            ts_pred_history=ts_pred_history,
            ts_true_history=ts_true_history,
            ts_pred_future=ts_pred_future,
            ts_past_k_history=ts_past_k_history,
            ts_past_k_future=ts_past_k_future,
            covariates_history=covariates_history,
            covariates_future=covariates_future,
            cat_covariates_history=cat_covariates_history,
            cat_covariates_future=cat_covariates_future,
            position_ids_history=position_ids_history,
            position_ids_future=position_ids_future,
        )
        
        ts_test = ts_test * ts_std + ts_mean
        return ts_test

    def rbf_kernel(
        self,
        x_1,
        x_2,
        width: float,
        variance: float,
        **kwargs
    ):
        '''
        
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

        K_15 = (1 + sqrt_3_d) * torch.exp(-sqrt_3_d)
        K_25 = (1 + sqrt_5_d + (5/3) * (dist**2 / width**2)) * torch.exp(-sqrt_5_d)

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

    def linear_kernel(
        self,
        x_1: Float[torch.Tensor, "batch num_examples num_features"],
        x_2: Float[torch.Tensor, "batch num_examples num_features"],
        variance: float,
        **kwargs
    ) -> Float[torch.Tensor, "batch num_examples num_examples"]:
        '''

        '''
        if type(variance) == torch.Tensor:
            variance = variance.reshape(-1, 1, 1)

        return variance * torch.matmul(x_1, x_2.transpose(1, 2))

    def gaussian_process_estimator(
        self,
        ts_pred_history,
        ts_true_history,
        ts_pred_future,
        ts_past_k_history,
        ts_past_k_future,
        covariates_history,
        covariates_future,
        cat_covariates_history,
        cat_covariates_future,
        position_ids_history,
        position_ids_future,
        **kwargs
    ):
        '''
            time series
            past_k
            temporal covariates
            categorical covariates
            positions

        '''
        # ts_weight = kwargs.get("ts_weight")
        # covariate_weight = kwargs.get("covariate_weight")
        # position_weight = kwargs.get("position_weight")

        # assert type(ts_weight) == type(covariate_weight) == type(position_weight)

        # if type(ts_weight) == torch.Tensor:
        #     ts_weight = ts_weight.reshape(-1, 1, 1)
        #     covariate_weight = covariate_weight.reshape(-1, 1, 1)
        #     position_weight = position_weight.reshape(-1, 1, 1)

        bsz = ts_pred_history.shape[0]
        # ts kernel params
        ts_kwargs = kwargs.get("ts_kwargs")
        # covariate kernel params
        covariate_kwargs = kwargs.get("covariate_kwargs")
        cat_covariate_kwargs = kwargs.get("cat_covariate_kwargs")
        # position kernel params
        position_kwargs = kwargs.get("position_kwargs")

        ts_pred_history = torch.cat([ts_pred_history, ts_past_k_history], dim=-1)
        ts_pred_future = torch.cat([ts_pred_future, ts_past_k_future], dim=-1)

        K = self.rbf_kernel(ts_pred_history, ts_pred_history, **ts_kwargs.get("rbf"))
        K += self.linear_kernel(ts_pred_history, ts_pred_history, **ts_kwargs.get("linear"))
        K += self.rbf_kernel(covariates_history, covariates_history, **covariate_kwargs.get("rbf"))
        K += self.linear_kernel(covariates_history, covariates_history, **covariate_kwargs.get("linear"))
        K += self.periodic_kernel(position_ids_history, position_ids_history, **position_kwargs.get("periodic"))


        K_s = self.rbf_kernel(ts_pred_history, ts_pred_future, **ts_kwargs.get("rbf"))
        K_s += self.linear_kernel(ts_pred_history, ts_pred_future, **ts_kwargs.get("linear"))
        K_s += self.rbf_kernel(covariates_history, covariates_future, **covariate_kwargs.get("rbf"))
        K_s += self.linear_kernel(covariates_history, covariates_future, **covariate_kwargs.get("linear"))
        K_s += self.periodic_kernel(position_ids_history, position_ids_future, **position_kwargs.get("periodic"))

        K_ss = self.rbf_kernel(ts_pred_future, ts_pred_future, **ts_kwargs.get("rbf"))
        K_ss += self.linear_kernel(ts_pred_future, ts_pred_future, **ts_kwargs.get("linear"))
        K_ss += self.rbf_kernel(covariates_future, covariates_future, **covariate_kwargs.get("rbf"))
        K_ss += self.linear_kernel(covariates_future, covariates_future, **covariate_kwargs.get("linear"))
        K_ss += self.periodic_kernel(position_ids_future, position_ids_future, **position_kwargs.get("periodic"))

        if self.has_cat_covariates:
            K += self.rbf_kernel(cat_covariates_history, cat_covariates_history, **cat_covariate_kwargs.get("rbf"))
            K_s += self.rbf_kernel(cat_covariates_history, cat_covariates_future, **cat_covariate_kwargs.get("rbf"))
            K_ss += self.rbf_kernel(cat_covariates_future, cat_covariates_future, **cat_covariate_kwargs.get("rbf"))

        K_reg = K + torch.eye(K.shape[1], device=K.device).unsqueeze(0) + self.eps 

        K_inv, info = torch.linalg.cholesky_ex(K_reg)

        if torch.any(info != 0):
            warnings.warn("Warning: K_reg is not positive definite! Increasing jitter...")

            jitter = 1e-4

            K_reg = K + jitter * torch.eye(K.shape[1], device=K.device).unsqueeze(0)

            try:
                L = torch.linalg.cholesky(K_reg)
            except:
                warnings.warn("Error: K_reg is still not invertible! Returning large values...")
                print("Warning: K_reg is not positive definite! Increasing jitter...")
                y_mean = torch.full((bsz, self.prediction_length, 1), 1e6, device=K.device)
                y_cov = torch.full((bsz, self.prediction_length, self.prediction_length), 1e6, device=K.device)
                return y_mean, y_cov
            

        alpha = torch.cholesky_solve(ts_true_history, K_inv)

        y_mean = torch.bmm(K_s.transpose(1,2), alpha)

        v = torch.cholesky_solve(K_s, K_inv)

        y_cov = K_ss - torch.bmm(K_s.transpose(1,2), v)

        return y_mean, y_cov

    def gradient_boosting_estimator(
        self,
        ts_pred_history,
        ts_true_history,
        ts_pred_future,
        ts_past_k_history,
        ts_past_k_future,
        covariates_history,
        covariates_future,
        cat_covariates_history,
        cat_covariates_future,
        position_ids_history,
        position_ids_future,
    ) -> Float[torch.Tensor, "batch num_test_examples patch_size"]:
        '''
            batch kernel regression
        '''
        x_train = torch.cat([ts_pred_history, ts_past_k_history, covariates_history, position_ids_history], dim=-1)
        x_test = torch.cat([ts_pred_future, ts_past_k_future, covariates_future, position_ids_future], dim=-1)

        y_train = ts_true_history

        if self.has_cat_covariates:
            x_train = torch.cat([x_train, cat_covariates_history], dim=-1)
            x_test = torch.cat([x_test, cat_covariates_future], dim=-1)

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