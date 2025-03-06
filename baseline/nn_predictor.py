from dataclasses import dataclass, field
from jaxtyping import Bool, Float, Int
from typing import List, Optional

import h5py
import logging
import numpy as np
import torch

from cuml import KernelRidge
from gluonts.itertools import batcher
from gluonts.model import Forecast
from gluonts.model.forecast import SampleForecast
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .nbeatx import NBEATSxForecastor
from .tide import TiDEForecastor

class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()
    
gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)

class NNPredictor:
    def __init__(
        self,
        model_name: str,
        batch_size: int,
        prediction_length: int,
        seasonality: int,
        use_covariates: bool,
        save_dir: str = None,
        model_config: str = None,
        test_run: bool = False,
        ds_config: dict = None,
        *args,
        **kwargs,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.prediction_length = prediction_length
        self.save_dir = save_dir
        self.model_config = model_config
        self.test_run = test_run
        self.ds_config = ds_config
        self.seasonality = seasonality
        self.use_covariates = use_covariates

        self.smaller_context = ds_config["smaller_context"]
        self.context_length = ds_config["context_length"]
        self.rolling_window_size = ds_config["rolling_window_size"]
        self.num_past_k = ds_config["num_past_k"]
        self.use_fixed_history = ds_config["use_fixed_history"]
        self.use_positions = ds_config["use_positions"]
        self.pos_dims = ds_config["pos_dims"]
        self.smallest_history = ds_config["smallest_history"]

    def save_predictions(
        self,
        save_tensors,
    ):
        with h5py.File(self.save_dir, "a") as f:
            if self.model_config in f:
                group = f[self.model_config]
                for key in save_tensors:
                    self.add_data(group, key, save_tensors[key])
            else:
                raise ValueError(f"{self.model_config} group not created!")
    
    def add_data(
        self,
        group: h5py.Group,
        key: str,
        data: Float[np.ndarray, "batch time tgt"],
    ):
        if key not in group:
            group.create_dataset(
                key, 
                data=data,
                maxshape=(None, *data.shape[1:]),
                chunks=True
            )
        else:
            dataset = group[key]
            dataset.resize((dataset.shape[0] + data.shape[0]), axis=0)
            dataset[-data.shape[0]:] = data  

    def _predict_example(
        self,
        past_target: Float[torch.Tensor, "context_length"],
        covariates: Float[torch.Tensor, "num_covariates total_length"],
    ):
        '''

        '''
        past_target = torch.Tensor(past_target).cuda().reshape(-1)
        covariates = torch.Tensor(covariates).cuda().T

        ts_mean = past_target.mean(dim=0, keepdim=True)
        ts_std = past_target.std(dim=0, keepdim=True) + 1e-6

        cov_mean = covariates[:self.context_length].mean(dim=0, keepdim=True)
        cov_std = covariates[:self.context_length].std(dim=0, keepdim=True) + 1e-6

        past_target = (past_target - ts_mean) / ts_std
        covariates = (covariates - cov_mean) / cov_std 

        if "nbeatsx" in self.model_name.lower():
            model = NBEATSxForecastor(
                past_target=past_target,
                covariates=covariates,
                ds_config=self.ds_config
            )
        elif "tide" in self.model_name.lower():
            model = TiDEForecastor(
                past_target=past_target,
                covariates=covariates,
                ds_config=self.ds_config
            )
        else:
            raise ValueError(f"{self.model_name} not configured")

        model.fit(n_trials=16)

        y_test = model.predict() # 1 x prediction_length

        print(y_test.shape)
    
        y_test = (y_test.squeeze(0) * ts_std + ts_mean).unsqueeze(0)

        return y_test

    def _predict_batch(
        self,
        batch: List[dict]
    ) -> Float[torch.Tensor, "bsz quantiles prediction_length"]:
        '''
            Runs baseline model
            Use history + covariates to train model and predict
                - autoregressively
                - parallely
            return tensor of shape (num_samples, prediction_length, target_dim)
        '''
        results = Parallel(n_jobs=16)(delayed(self._predict_example)(past_target=entry["target"], covariates=entry["feat_dynamic_real"]) for entry in tqdm(batch, desc="Baseline (regression): ", leave=False))

        # results = []
        # for entry in tqdm(batch, desc="Baseline (regression): ", leave=False):
        #     y = self._predict_example(past_target=entry["target"], covariates=entry["feat_dynamic_real"])
        #     results.append(y)

        y_test = torch.stack(results)

        assert y_test.shape[1:] == (1, self.prediction_length)
        return y_test
    
    def predict(self, test_data_input) -> List[Forecast]:
        while True:
            try: 
                forecast_outputs = []
                for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
                    forecast = self._predict_batch(batch)
                    forecast_outputs.append(forecast.cpu().numpy())
                forecast_outputs = np.concatenate(forecast_outputs)
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"OutOfMemoryError at batch_size {self.batch_size}, reducing to {self.batch_size // 2}"
                )
                self.batch_size //= 2
        
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples = item, start_date = forecast_start_date)
            )
        
        return forecasts
                