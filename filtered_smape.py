import numpy as np 

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional

from gluonts.ev.metrics import DirectMetric, BaseMetricDefinition
from gluonts.ev.aggregations import Mean

def symmetric_absolute_percentage_error(
    data: Dict[str, np.ndarray], forecast_type: str, atol: float = 1e-3
) -> np.ndarray:
    label = data["label"]
    forecast = data[forecast_type]

    abs_label = np.abs(label)
    abs_forecast = np.abs(forecast)
    denom = abs_label + abs_forecast

    ignore_mask = np.isclose(label, 0.0, atol=atol)

    valid_mask = (denom != 0) & (~ignore_mask)

    smape = np.zeros_like(label, dtype=np.float32)
    smape[valid_mask] = 2.0 * np.abs(label[valid_mask] - forecast[valid_mask]) / denom[valid_mask]

    return smape

@dataclass
class filteredSMAPE(BaseMetricDefinition):
    """Symmetric Mean Absolute Percentage Error"""

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        return DirectMetric(
            name=f"filtered-sMAPE[{self.forecast_type}]",
            stat=partial(
                symmetric_absolute_percentage_error,
                forecast_type=self.forecast_type,
            ),
            aggregate=Mean(axis=axis),
        )


smape = filteredSMAPE()