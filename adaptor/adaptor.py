import h5py
import math
import torch

from jaxtyping import Bool, Float, Int
from typing import Any, Dict, Generator, List, Optional

import cupy 

VALIDATE_FEATURES={
    0: {
        "features": ["past_k", "covariates", "position"],
    },
    1: {
        "features": ["covariates", "positions"],
    },
    2: {
        "features": ["past_k", "covariates"],
    },
    3: {
        "features": ["covariates"],
    }
}

VALIDATE_MODELS={
    0: {
        "model": "kernel_regression",
        "kernel": "rbf",
    },
    1: {
        "model": "kernel_regression",
        "kernel": "linear",
    },
    2: {
        "model": "kernel_regression",
        "kernel": "polynomial",
    },
    3: {
        "model": "nadaraya_watson",
    },
    4: {
        "model": "gaussian_process",
    }
}

class Adaptor:
    ''''
        pass normalized features y_i, x_i, p_i
    '''
    def __init__(
        self,
        x_train: Float["batch", "num_train_examples", "num_features"],
        x_test: Float["batch", "num_test_examples", "num_features"],
        y_train: Float["batch", "num_train_examples", "patch_size"],
        y_test: Float["batch", "num_test_examples", "patch_size"],
        features: Dict[str, Float["batch", "num_total_examples", "num_features"]],
        ds_config: Dict[str, Any],
    ):
        self.x_train = x_train
        self.x_test = x_test
        self.features = features
        self.ds_config = ds_config

    def predict_with_validation(self):
        '''
            Cross validate using y_{H-F:H}
                * adaptor method
                * features

            Train the final model with all training data
        '''
        x_train = self.x_train[:, :-self.ds_config.prediction_length]
        x_val = self.x_train[:, -self.ds_config.prediction_length:]
        y_train = self.y_train[:, :-self.ds_config.prediction_length]
        y_test = self.y_train[:, -self.ds_config.prediction_length:]

    def evaluate_adaptors(self, y_true):
        for a, adaptor in VALIDATE_MODELS.items():
            for f, feature in VALIDATE_FEATURES.items():
                y_test = self.evaluate_adaptor(feature, adaptor)
                # compute results
                # log results 
                # check with best performing


    def evaluate_adaptor(self, features, model_args):

        x_train, x_test = self.append_features(self, features)
        adaptor = self.select_adaptor(**model_args)
        y_test = adaptor(x_train, self.y_train, x_test, **model_args)

        return y_test

    def select_adaptor(self, **args):
        if args.model == "kernel_regression":
            return self.kernel_regression_estimator
        elif args.model == "nadaraya_watson":
            return self.nadaraya_watson_kernel_estimator
        elif args.model == "gaussian_process":
            return self.eval_gaussian_process_estimator
        else:
            raise ValueError(f"Invalid adaptor: {args.model}")
           
    def append_features(
        self,
        features: List[str],
    ):
        '''
            x_train: time series y_i
            x_test: time series y_i

            features: 
                position: 
                past_k:
                covariates: 
        '''
        if "past_k" in features:
            self.concat_past_k()
    
        if "covariates" in features:
            self.concat_covariates()

        if "position" in features:
            self.concat_position()


    def eval_kernel_regression_estimator(self, x_train, y_train, x_test):
        '''
            x_train: batch x num_train_examples x num_features
            x_test: batch x num_test_examples x num_features
            y_train: batch x num_train_examples x patch_size
        '''

        y_test = self.kernel_regression_estimator(x_train, y_train, x_test)

        y_test = y_test * self.y_std + self.y_mean

        return y_test
    
    def eval_nadaraya_watson_kernel_estimator(self, x_train, y_train, x_test):
        '''
            x_train: batch x num_train_examples x num_features
            x_test: batch x num_test_examples x num_features
            y_train: batch x num_train_examples x patch_size
        '''

        y_test = self.nadaraya_watson_kernel_estimator(x_train, y_train, x_test)

        y_test = y_test * self.y_std + self.y_mean

        return y_test
        

    def eval_gaussian_process_estimator(
        self, 
        x_train: Float["batch", "num_train_examples", "num_features"], 
        y_train: Float["batch", "num_train_examples", "patch_size"],
        x_test: Float["batch", "num_test_examples", "num_features"],
        variance_threshold: List[Int]):
        '''
            x_train: batch x num_train_examples x num_features
            x_test: batch x num_test_examples x num_features
            y_train: batch x num_train_examples x patch_size
        '''
        y_test_no_filtering = self.gaussian_process_estimator(x_train, y_train, x_test, variance_based_filtering=False)

        y_test_with_filtering, covariance = self.gaussian_process_estimator(x_train, y_train, x_test, variance_based_filtering=True)

        variance = torch.diagonal(covariance, dim1=1, dim2=2).unsqueeze(-1) 

        variance_filtered_y_test = []
        for threshold in variance_threshold:
            y_test = torch.where(variance > threshold, self.fm_y_test, y_test_no_filtering)
            variance_filtered_y_test.append(y_test)
        
        return y_test_no_filtering, variance_filtered_y_test

    def concat_position(
        self,
    ):
        position_test = self.features["position"][:, -self.ds_config.prediction_length:]
        position_train = self.features["position"][:, :-self.ds_config.prediction_length]

        x_train = torch.cat([self.x_train, position_train], dim=2)
        x_test = torch.cat([self.x_test, position_test], dim=2)
        return x_train, x_test

    def concat_past_k(
        self,
    ):
        past_k_test = self.features["past_k"][:, -self.ds_config.prediction_length:]
        past_k_train = self.features["past_k"][:, :-self.ds_config.prediction_length]

        x_train = torch.cat([self.x_train, past_k_train], dim=2)
        x_test = torch.cat([self.x_test, past_k_test], dim=2)
        return x_train, x_test

    def concat_covariates(
        self,
    ):
        covariates_test = self.features["covariates"][:, -self.ds_config.prediction_length:]
        covariates_train = self.features["covariates"][:, :-self.ds_config.prediction_length]

        x_train = torch.cat([self.x_train, covariates_train], dim=2)
        x_test = torch.cat([self.x_test, covariates_test], dim=2)
        return x_train, x_test
        
    def rbf_kernel(self):
        '''
            compute batch rbf kernel

        '''

        pass

    def linear_kernel(self):
        '''
            compute batch linear kernel

        '''
        pass

    def kernel_regression_estimator(self, x_train, y_train, x_test):
        '''
            batch kernel regression

        '''
        pass

    def nadaraya_watson_kernel_estimator(self):
        '''
            batch nadaraya watson kernel estimator

        '''
        pass

    def gaussian_process_estimator(self):
        '''
            batch gaussian process estimator

        '''
        pass