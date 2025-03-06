from datasets import Dataset, DatasetInfo, Features, Sequence, Value
import numpy as np
import os
import pandas as pd
import torch 

from dotenv import load_dotenv
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

load_dotenv()
storage_path = Path(os.getenv("RAW_DATA"))
vldb_path=Path(os.getenv("VLDB_BENCH"))
save_path = Path(os.getenv("GIFT_EVAL"))

def load_epf_datasets():
    datasets=["BE", "DE", "FR", "NP", "PJM"]
       
    for dataset in datasets:
        data=pd.read_csv(str(storage_path / (dataset+".csv")))
        target_col=""
        cov_cols=[]
        date_col=""
        for col in data.columns:
            if "price" in col.lower():
                target_col=col
            elif "date" in col.lower() or "unnamed" in col.lower():
                date_col=col
            else:
                cov_cols.append(col)
        item_id=dataset
        freq="h"
        start=pd.to_datetime(data[date_col][0])
        target=data[target_col].to_numpy()
        feat_dynamic_real=data[cov_cols].to_numpy().T
        hfdataset = Dataset.from_dict({
            "item_id": [item_id],
            "start": [start],
            "freq": [freq],
            "target": [target],
            "feat_dynamic_real": [feat_dynamic_real],
        }, info=DatasetInfo(
                description="",
                citation='Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". Applied Energy 2021; 293:116983. https://doi.org/10.1016/j.apenergy.2021.116983',
                homepage="",
                license="",
                features=Features({
                    "item_id": Value("string"),
                    "start": Value("timestamp[s]"),
                    "freq": Value("string"),
                    "target": Sequence(Value("float32")),
                    "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=2),
                }),
                dataset_name=dataset,
                builder_name="generator",
        ))
        hfdataset.save_to_disk(str(save_path / dataset))

def load_bench_vldb_datasets():
    datasets=["air_quality", "bafu", "chlorine", "climate", "electricity", "drift", "temp", "meteo"]
    dataset_frequencies = {
        "air_quality": "H",   # Hourly
        "bafu": "30T",        # 30 minutes
        "chlorine": "5T",     # 5 minutes
        "climate": "M",       # Monthly
        "electricity": "15T", # 15 minutes
        "drift": "6H",        # 6 hours
        "temp": "D",          # Daily
        "meteo": "10T"        # 10 minutes
    }
       
    for dataset in datasets:
        data=np.loadtxt(str(vldb_path / dataset / (dataset+"_normal.txt")))
        print(data.shape) # serties len x num series

        # find covariates for ecah adataset and create covariates array 
        correlation_matrix = np.corrcoef(data, rowvar=False)

        selected_covariates = []
        num_series = correlation_matrix.shape[0]
        series_len = data.shape[0]
        num_covariates = 4 if num_series <= 10 else 10
        
        for i in range(num_series):
            correlations = np.abs(correlation_matrix[i]) 
            correlations[i] = -np.inf  

            top_covariates = np.argsort(correlations)[-num_covariates:]

            print(dataset, np.sort(correlations)[-num_covariates:])

            selected_covariates.append(data[:, top_covariates])

        continue

        feat_dynamic_real = np.array(selected_covariates) # (num_series, series_len, num_covariates)

        target = data.astype(np.float32) # (series_len x num_series)
        feat_dynamic_real=feat_dynamic_real.astype(np.float32) # (num_series, series_len, num_covariates)

        item_id = dataset
        freq = dataset_frequencies[dataset]
        start = pd.to_datetime("2000-01-01")

        hfdataset = Dataset.from_dict({
            "item_id": [item_id] * num_series,
            "start": [start] * num_series,
            "freq": [freq] * num_series,
            "target": [target[:, i] for i in range(num_series)], 
            "feat_dynamic_real": [feat_dynamic_real[i].T for i in range(num_series)], 
        }, info=DatasetInfo(
            description="",
            citation='Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". Applied Energy 2021; 293:116983. https://doi.org/10.1016/j.apenergy.2021.116983',
            homepage="",
            license="",
            features = Features({
                "item_id": Value("string"),
                "start": Value("timestamp[s]"),
                "freq": Value("string"),
                "target": Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None),  # univariate (series_len, )
                "feat_dynamic_real": Sequence(feature=Sequence(feature=Value("float32"), id=None), length=-1) # covariates (num_covariates, series_len)
            }),
            dataset_name=dataset,
            builder_name="generator",
        ))

        hfdataset.save_to_disk(str(save_path / dataset))

def load_covid_datasets():
    '''
        Following the setting authors used: https://arxiv.org/pdf/2107.10397
        skip first 44 days
        Total length after removing missing data: 376 days 
        last 140 days used as test data
    '''
    dataset="CLEANED_35_Updated"
       
    data=pd.read_csv(str(storage_path / (dataset+".csv")), index_col="SEQNUM")
    data.drop([i for i in range(1,45)], inplace=True)
    target_col="deathIncrease"
    cov_cols=["hospitalizedCurrently", "inIcuCurrently", "hospitalizedCumulative", "onVentilatorCurrently"]
    date_col="date"
    item_id="covid-cases-US"
    freq="D"
    start=pd.to_datetime(data[date_col].tolist()[0])
    print(start)
    target=data[target_col].to_numpy()
    feat_dynamic_real=data[cov_cols].to_numpy().T
    hfdataset = Dataset.from_dict({
        "item_id": [item_id],
        "start": [start],
        "freq": [freq],
        "target": [target],
        "feat_dynamic_real": [feat_dynamic_real],
}, info=DatasetInfo(
            description="saving as covid-national",
            citation='Toutiaee, Mohammadhossein, et al. "Improving COVID-19 forecasting using exogenous variables." arXiv preprint arXiv:2107.10397 (2021).',
            homepage="",
            license="",
            features=Features({
                "item_id": Value("string"),
                "start": Value("timestamp[s]"),
                "freq": Value("string"),
                "target": Sequence(Value("float32")),
                "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=len(cov_cols)),
            }),
            dataset_name=dataset,
            builder_name="generator",
    ))
    dataset="covid-national"
    print(f'Saving dataset to {str(save_path / dataset)}')
    hfdataset.save_to_disk(str(save_path / dataset))

def load_bike_sharing_datasets():
    dataset="bike_sharing"
    data=pd.read_csv(str(storage_path / (dataset+".csv")))

    target_col="cnt"
    cov_cols=['season', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
    item_id="bike_sharing"

    one_hot_cols = ["season", "weekday", "weathersit"] 
    encoder = OneHotEncoder(sparse_output=False, drop="first")  
    one_hot_encoded = encoder.fit_transform(data[one_hot_cols])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(one_hot_cols))

    data = data.drop(columns=one_hot_cols).reset_index(drop=True)
    data = pd.concat([data, one_hot_df], axis=1)

    cov_cols = [col for col in cov_cols if col not in one_hot_cols]
    cov_cols.extend(one_hot_df.columns)

    freq="h"
    start=pd.to_datetime(data["dteday"][0])
    target=data[target_col].to_numpy()
    feat_dynamic_real=data[cov_cols].to_numpy().T
    hfdataset = Dataset.from_dict({
        "item_id": [item_id],
        "start": [start],
        "freq": [freq],
        "target": [target],
        "feat_dynamic_real": [feat_dynamic_real],
    }, info=DatasetInfo(
            description="",
            citation='Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron. "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark". Applied Energy 2021; 293:116983. https://doi.org/10.1016/j.apenergy.2021.116983',
            homepage="",
            license="",
            features=Features({
                "item_id": Value("string"),
                "start": Value("timestamp[s]"),
                "freq": Value("string"),
                "target": Sequence(Value("float32")),
                "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=len(cov_cols)),
            }),
            dataset_name=dataset,
            builder_name="generator",
    ))
    hfdataset.save_to_disk(str(save_path / dataset))

def load_general_datasets(dataset, target_col, cov_cols, date_col, freq, citation):
    '''
    '''
       
    data=pd.read_csv(str(storage_path / (dataset+".csv")), index_col=0)
    data.drop([i for i in range(1,45)], inplace=True)
    data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
    data = data.sort_values(by=date_col)
    item_id=dataset
    freq=freq
    start=pd.to_datetime(data[date_col].tolist()[0])
    print(start)
    target=data[target_col].to_numpy()
    feat_dynamic_real=data[cov_cols].to_numpy().T
    hfdataset = Dataset.from_dict({
        "item_id": [item_id],
        "start": [start],
        "freq": [freq],
        "target": [target],
        "feat_dynamic_real": [feat_dynamic_real],
}, info=DatasetInfo(
            description="",
            citation=citation,
            homepage="",
            license="",
            features=Features({
                "item_id": Value("string"),
                "start": Value("timestamp[s]"),
                "freq": Value("string"),
                "target": Sequence(Value("float32")),
                "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=len(cov_cols)),
            }),
            dataset_name=dataset,
            builder_name="generator",
    ))
    print(f'Saving dataset to {str(save_path / dataset)}')
    hfdataset.save_to_disk(str(save_path / dataset))

def load_sines_dataset():
    '''
    '''
    freq = 24
    amp = 10
    num_values = 10000
    i = torch.arange(0, num_values, dtype=torch.float32).reshape(-1, 1)
    y = amp * torch.sin(2 * torch.pi * i / freq)
    div = torch.tensor([2, 5, 10]).reshape(1, -1)
    cov = y / div

    item_id="sines"
    freq="h"
    start=pd.to_datetime("2008-07-15")
    print(start)
    target=y.numpy()
    feat_dynamic_real=cov.numpy().T
    hfdataset = Dataset.from_dict({
        "item_id": [item_id],
        "start": [start],
        "freq": [freq],
        "target": [target],
        "feat_dynamic_real": [feat_dynamic_real],
}, info=DatasetInfo(
            description="",
            citation="test dataset",
            homepage="",
            license="",
            features=Features({
                "item_id": Value("string"),
                "start": Value("timestamp[s]"),
                "freq": Value("string"),
                "target": Sequence(Value("float32")),
                "feat_dynamic_real": Sequence(Sequence(Value("float32")), length=cov.shape[1]),
            }),
            dataset_name="sines",
            builder_name="generator",
    ))
    print(f'Saving dataset to {str(save_path / "sines")}')
    hfdataset.save_to_disk(str(save_path / "sines"))


if __name__ == "__main__":
    # load_epf_datasets()
    # load_bench_vldb_datasets()
    # load_covid_datasets()
    
    # # electric_consumption
    # load_general_datasets(dataset="electric_consumption", target_col="Consumption", cov_cols=["Homestead_maxtempC", "Homestead_mintempC", "Homestead_DewPointC", "Homestead_FeelsLikeC", "Homestead_HeatIndexC", "Homestead_WindChillC", "Homestead_WindGustKmph", "Homestead_cloudcover", "Homestead_humidity", "Homestead_precipMM", "Homestead_pressure", "Homestead_tempC", "Homestead_visibility", "Homestead_winddirDegree", "Homestead_windspeedKmph"], date_col="Date", freq="h", citation="https://www.kaggle.com/datasets/unajtheb/homesteadus-electricity-consumption")

    # load_general_datasets(dataset="electric_demand",  target_col="Electric_demand", cov_cols=['Temperature', 'GHI', 'Humidity', 'Season', 'PV_production', 'Day_of_the_week', 'Wind_speed', 'DHI', 'Wind_production', 'DNI'], date_col="Time", freq="5T", citation="Rojas Ortega, Sebastian; Castro-Correa, Paola; Sepúlveda-Mora, Sergio; Castro-Correa, Jhon (2023), “Renewable Energy and Electricity Demand Time Series Dataset with Exogenous Variables at 5-minute Interval”, Mendeley Data, V1, doi: 10.17632/fdfftr3tc2.1")

    # load_bike_sharing_datasets()

    load_bench_vldb_datasets()
        
        