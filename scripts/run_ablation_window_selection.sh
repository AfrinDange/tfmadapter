# model_name="amazon/chronos-bolt-base"
# model="chronos"

model_name="Salesforce/moirai-1.1-R-base"
model="moirai"

# model_name="google/timesfm-2.0-500m-pytorch"
# model="timesfm"

device=4

# "temp bafu electricity air_quality climate meteo NP PJM BE DE FR drift chlorine"

CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
    --model_name $model_name \
    --dataset_names "temp bafu electricity air_quality climate meteo NP PJM BE DE FR drift chlorine" \
    --batch_size 1024 \
    --dataset_config dataset_config2 \
    --num_past_k 4 \
    --pos_dims 16 \
    --use_positions \
    --model_config "${model}_adaptor_with_cov_random_window" \
    --folds 1 \
    --adaptor_method "gaussian_process" \
    --log_subdir "rebuttal-ablation" \
    --random_window_selection 

CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
    --model_name $model_name \
    --dataset_names "meteo drift bafu air_quality DE NP PJM BE" \
    --batch_size 1024 \
    --dataset_config dataset_config2 \
    --num_past_k 4 \
    --pos_dims 16 \
    --use_positions \
    --model_config "${model}_adaptor_with_cov_latest_window" \
    --folds 1 \
    --adaptor_method "gaussian_process" \
    --log_subdir "rebuttal-ablation" \
    --no_window_selection 

CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
    --model_name $model_name \
    --dataset_names "temp bafu electricity air_quality climate meteo NP PJM BE DE FR drift chlorine" \
    --batch_size 1024 \
    --dataset_config dataset_config2 \
    --num_past_k 4 \
    --pos_dims 16 \
    --use_positions \
    --model_config "${model}_adaptor_with_cov_zscore_selection" \
    --folds 1 \
    --adaptor_method "gaussian_process" \
    --log_subdir "rebuttal-ablation" 
