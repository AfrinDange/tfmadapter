import math
import numpy as np 
import optuna
import torch 

from jaxtyping import Float
from neuralforecast.models import TimeXer
from torch.nn.functional import l1_loss
from tqdm import trange

class TimeXerForecastor():
    def __init__(
        self,
        past_target: Float[torch.Tensor, "entire_context"],
        covariates: Float[torch.Tensor, "total_length num_covariates"],
        ds_config
    ):
        self.entire_context = ds_config["context_length"]
        self.context_length = ds_config["smallest_history"]
        self.prediction_length = ds_config["prediction_length"]
        self.batch_size = 16

        self.futr_exog_list = [f'cov_{i}' for i in range(covariates.shape[-1])]
 
        self.x_train = past_target[:-self.prediction_length].unfold(dimension=0, size=self.context_length, step=1) # examples x history
        self.futr_cov_train = covariates[:self.entire_context].unfold(dimension=0, size=self.context_length+self.prediction_length, step=1).transpose(1,2) # examples x history+forecast x num_covariates

        self.y_train = past_target[self.context_length:].unfold(dimension=0, size=self.prediction_length, step=1) # examples x forecast
        
        self.x_test = past_target[-self.context_length:].unsqueeze(0) # 1 x history
        self.futr_cov_test = covariates[-self.context_length-self.prediction_length:].unsqueeze(0) # 1 x history+forecast x num_covariates

        assert self.x_train.shape[0] == self.futr_cov_train.shape[0]
        assert self.y_train.shape[0] == self.futr_cov_train.shape[0]

        self.num_train_examples = self.x_train.shape[0] - 5
        self.num_val_examples = 5
        self.model = None

    def _train_batch(self, model, optimizer):
        total_loss = 0.0
        model.train()

        for b in range(0, self.num_train_examples, self.batch_size):
            insample_y = self.x_train[b:b+self.batch_size]
            futr_exog = self.futr_cov_train[b:b+self.batch_size]
            y_true = self.y_train[b:b+self.batch_size]

            optimizer.zero_grad()
            y_pred = model({
                "insample_y": insample_y.unsqueeze(-1),
                "futr_exog": futr_exog.unsqueeze(1),
            }).squeeze(-1)

            loss = l1_loss(y_pred, y_true) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / math.ceil(self.num_train_examples / self.batch_size)
    
    def _validate(self, model):
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for b in range(self.num_train_examples, self.num_train_examples+self.num_val_examples, self.batch_size):
                insample_y = self.x_train[b:b+self.batch_size]
                insample_mask = torch.ones_like(insample_y, dtype=torch.bool)
                futr_exog = self.futr_cov_train[b:b+self.batch_size]
                y_true = self.y_train[b:b+self.batch_size]

                y_pred = model({
                    "insample_y": insample_y.unsqueeze(-1),
                    "futr_exog": futr_exog.unsqueeze(1),
                }).squeeze(-1)

                loss = l1_loss(y_pred, y_true)
                total_loss += loss.item()

        return total_loss / math.ceil(self.num_val_examples / self.batch_size)
    
    def _objective(self, trial):
        dropout_prob_theta = trial.suggest_float("dropout_prob_theta", low=0.05, high=0.3, log=True)
        learning_rate = trial.suggest_float("learning_rate", low=1e-4, high=5e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128])

        model = TimeXer(
            h=self.prediction_length,
            input_size=self.context_length,
            n_series=1,
            futr_exog_list=self.futr_exog_list,
            patch_len=16,    
            hidden_size=hidden_size,
            n_heads=2,            
            e_layers=1,           
            d_ff=64,         
            dropout=dropout_prob_theta,
            use_norm=True,
            learning_rate=learning_rate,
            max_steps=1000,
            batch_size=self.batch_size,
            random_seed=42,
        ).cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(5):
            train_loss = self._train_batch(model, optimizer)
            val_loss = self._validate(model)

        return val_loss 
    
    def fit(self, n_trials=16):
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=n_trials)

        best_params = study.best_params
        
        self.model = TimeXer(
            h=self.prediction_length,
            input_size=self.context_length,
            n_series=1,
            futr_exog_list=self.futr_exog_list,
            patch_len=16,    
            hidden_size=best_params["hidden_size"],
            n_heads=2,            
            e_layers=1,           
            d_ff=64,         
            dropout=best_params["dropout_prob_theta"],
            use_norm=True,
            learning_rate=best_params["learning_rate"],
            max_steps=1000,
            batch_size=self.batch_size,
            random_seed=42,
        ).cuda()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=best_params["learning_rate"])

        for epoch in trange(50, desc="Training TimeXer"):
            train_loss = self._train_batch(self.model, optimizer)
            val_loss = self._validate(self.model)

            trange(50).set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Val Loss": f"{val_loss:.4f}"
            })

    def predict(self):
        if self.model is None:
            raise ValueError("Model is not trained. Call `fit()` first.")

        with torch.no_grad():
            forecast = self.model({
                "insample_y": self.x_test.unsqueeze(-1),
                "insample_mask": torch.ones_like(self.x_test, dtype=torch.bool),
                "futr_exog": self.futr_cov_test.unsqueeze(1),
                "hist_exog": None,
                "stat_exog": None,
            })

        return forecast
