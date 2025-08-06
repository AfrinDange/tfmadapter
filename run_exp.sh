# model_name="amazon/chronos-bolt-base"
# model_name="amazon/chronos-bolt-small"
# model="chronos"

model_name="Salesforce/moirai-1.1-R-base"
model="moirai-base"

# model_name="google/timesfm-2.0-500m-pytorch"
# model="timesfm"

device=7

# CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
#     --model_name $model_name \
#     --dataset_names "bike_sharing" \
#     --batch_size 1024 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config "${model}_univariate" \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --log_subdir "rebuttal-add-exp" \
#     --test_run

CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
    --model_name $model_name \
    --dataset_names "ETTh1 ETTh2 ETTm1 ETTm2" \
    --batch_size 1024 \
    --dataset_config dataset_config2 \
    --num_past_k 4 \
    --pos_dims 16 \
    --use_positions \
    --model_config "${model}_adapter_with_cov" \
    --folds 1 \
    --adaptor_method "gaussian_process" \
    --log_subdir "moirai_ablation_2.0"

# CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
#     --model_name $model_name \
#     --dataset_names "bike_sharing" \
#     --batch_size 1024 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config "${model}_adaptor_with_cov" \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --log_subdir "rebuttal-add-exp"

# CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
#     --model_name nbeatsx_baseline \
#     --dataset_names "DE PJM" \
#     --batch_size 1024 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config "nbeatsx_baseline" \
#     --folds 1 \
#     --log_subdir "rebuttal-add-exp" \
#     --adaptor_method "gaussian_process" \
#     --test_run

# CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
#     --model_name nbeatsxg_baseline \
#     --dataset_names "NP" \
#     --batch_size 1024 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config "nbeatsxg_baseline" \
#     --folds 1 \
#     --log_subdir "chronosx_comparison" \
#     --adaptor_method "gaussian_process"