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
from gluonts.model import evaluate_model
# from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.time_feature import get_seasonality

from gift_eval.data import Dataset
# from uni2ts.model.moirai import MoiraiForecast, MoiraiAdaptorForecast, MoiraiAdaptorExtendedForecast, MoiraiModule

from chronos import ChronosPredictor, ChronosAdaptorPredictor

# args


# metrics

# logger

# run eval

