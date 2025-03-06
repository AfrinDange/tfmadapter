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

from dataclasses import dataclass

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

class RegressorPredictor:
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
        past_target: Float[torch.Tensor, "context_length target_dim"],
        covariates: Float[torch.Tensor, "total_length num_covariates"],
        positions: Float[torch.Tensor, "total_length pos_dims"],
    ):
        '''

        '''
        #
        past_target = torch.Tensor(past_target).cuda().reshape(-1, 1)
        covariates = torch.Tensor(covariates).cuda().T
        positions = torch.Tensor(positions)

        past_k = torch.cat([
            past_target[:1].repeat(self.num_past_k, 1),
            past_target[:-1]
        ], dim=0).unfold(dimension=0, size=self.num_past_k, step=1).squeeze(-2)

        ts_mean = past_target.mean(dim=0, keepdim=True)
        ts_std = past_target.std(dim=0, keepdim=True) + 1e-6

        cov_mean = covariates.mean(dim=0, keepdim=True)
        cov_std = covariates.std(dim=0, keepdim=True) + 1e-6

        x_ = torch.cat([
            (covariates - cov_mean) / cov_std,
            positions
        ], dim=-1)
        x_train = torch.cat([(past_k - ts_mean) / ts_std, x_[:self.context_length]], dim=-1)
        x_test = x_[self.context_length:]
        x_test_past_t = (past_target[-self.num_past_k:] - ts_mean) / ts_std
        x_test_past_t = x_test_past_t.T # 1 x num_past_k
        y_train = (past_target - ts_mean) / ts_std

        regr_model = KernelRidge(kernel="rbf", alpha=1.0)

        regr_model.fit(x_train, y_train)

        y_test = torch.empty((self.prediction_length, 1), device=x_test.device)
        for i in range(self.prediction_length):
            y_test_i = torch.from_numpy(regr_model.predict(torch.cat([x_test_past_t, x_test[i:i+1]], dim=-1)).get()).to(device=x_test.device)

            x_test_past_t[:, :-1] = x_test_past_t[:, 1:].clone()
            x_test_past_t[:, -1] = y_test_i
        
            y_test[i:i+1] = y_test_i

        y_test = y_test * ts_std + ts_mean

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
        # examples
        num_covariates = batch[0]["feat_dynamic_real"].shape[0]
        mini_batch_size = len(batch)
        total_length = self.context_length + self.prediction_length

        if self.use_positions:
            positions = (torch.arange(self.seasonality, device="cuda")).type(torch.float32).repeat((total_length) // self.seasonality + 1)[-(total_length):]
            positions = positions.unsqueeze(1).repeat(1, self.pos_dims)
            dim_positions = torch.pow(torch.tensor([1e4]), 2 * (torch.arange(self.pos_dims) // 2)/self.pos_dims).unsqueeze(0).to("cuda")
            positions[:, ::2] = torch.sin(positions[:, ::2] / dim_positions[:, ::2])
            positions[:, 1::2] = torch.cos(positions[:, 1::2] / dim_positions[:, 1::2])


        results = Parallel(n_jobs=16)(delayed(self._predict_example)(past_target=entry["target"], covariates=entry["feat_dynamic_real"], positions=positions) for entry in tqdm(batch, desc="Baseline (regression): ", leave=False))

        # for entry in tqdm(batch, desc="Baseline (regression): ", leave=False):
        #     y = self._predict_example(past_target=entry["target"], covariates=entry["feat_dynamic_real"], positions=positions)
        #     results.append(y)

        y_test = torch.stack(results).transpose(1,2)

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
                