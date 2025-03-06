CUDA_VISIBLE_DEVICES=7 python3 evaluate_model.py \
    --model_name amazon/chronos-bolt-base \
    --dataset_names "NP BE DE FR PJM air_quality meteo temp electricity climate bafu" \
    --batch_size 1024 \
    --dataset_config dataset_config2 \
    --num_past_k 4 \
    --pos_dims 16 \
    --use_positions \
    --model_config chronos_adaptor__with_cov_gb \
    --folds 1 \
    --adaptor_method "xgboost" \
    --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates" \
    --log_subdir "gradient_boosting"

# CUDA_VISIBLE_DEVICES=6 python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "temp" \
#     --batch_size 512 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_univariate \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection "" &

# CUDA_VISIBLE_DEVICES=5 python3 evaluate_model.py \
#     --model_name Salesforce/moirai-1.1-R-base \
#     --dataset_names "temp" \
#     --batch_size 16 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config moirai_univariate \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection ""

# wait

# CUDA_VISIBLE_DEVICES=6 python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "temp" \
#     --batch_size 512 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_adaptor__with_cov__pos=16 \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates" \
#     --context_length 512 \
#     --smaller_context 416 \
#     --smallest_history 128 \
#     --test_run

# CUDA_VISIBLE_DEVICES=5 python3 evaluate_model.py \
#     --model_name Salesforce/moirai-1.1-R-base \
#     --dataset_names "air_quality meteo temp electricity climate bafu" \
#     --batch_size 16 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config moirai_with_covariate \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection ""

# wait

# # google/timesfm-2.0-500m-pytorch
# # Salesforce/moirai-1.1-R-base

# CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
#     --model_name google/timesfm-2.0-500m-pytorch \
#     --dataset_names "air_quality meteo temp electricity climate bafu" \
#     --batch_size 512 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config timesfm_adaptor_with_cov \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates" 

# CUDA_VISIBLE_DEVICES=6 python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "FR NP BE DE PJM" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_adaptor_with_cov__no_window_selection_3F_saved_v2 \
#     --no_window_selection \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --smallest_history 600 \
#     --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates" &

# CUDA_VISIBLE_DEVICES=7 python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "FR NP BE DE PJM" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_adaptor_with_cov__window_selection_3F_saved_v2 \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --smallest_history 168 \
#     --smaller_context 600 \
#     --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates"

# wait 
    
# CUDA_VISIBLE_DEVICES=0 python3 evaluate_model.py \
#     --model_name Salesforce/moirai-1.1-R-base \
#     --dataset_names "bike_sharing" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config moirai_GP_adaptor___with_cov__ \
#     --folds 1 \
#     --adaptor_method "gaussian_process"

# CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
#     --model_name google/timesfm-2.0-500m-pytorch \
#     --dataset_names "bike_sharing" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config timesfm_GP_adaptor_with_cov__ \
#     --folds 1 \
#     --adaptor_method "gaussian_process"

# #"air_quality bafu climate electricity temp meteo bike_sharing" \

# # CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
# #     --model_name amazon/chronos-bolt-base \
#     # --dataset_names "bike_sharing" \
# #     --batch_size 1024 \
# #     --dataset_config dataset_config_ablation \
# #     --num_past_k 4 \
# #     --pos_dims 16 \
# #     --use_positions \
# #     --model_config chronos_one_stage_adaptor_with_cov \
# #     --folds 1 \
# #     --adaptor_method "gaussian_process" \
# #     --test_run \
# #     --remove_pseudo_forecast_generator

# CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "bike_sharing" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_univariate \
#     --folds 1 \
#     --adaptor_method "" 

# CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
#     --model_name Salesforce/moirai-1.1-R-base \
#     --dataset_names "bike_sharing" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config moirai_univariate \
#     --folds 1 \
#     --adaptor_method "" 

# # CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
# #     --model_name Salesforce/moirai-1.1-R-base \
# #     --dataset_names "bike_sharing" \
# #     --batch_size 728 \
# #     --dataset_config dataset_config2 \
# #     --num_past_k 4 \
# #     --pos_dims 16 \
# #     --use_positions \
# #     --model_config moirai_with_covariate \
# #     --folds 1 \
# #     --adaptor_method "" 

# CUDA_VISIBLE_DEVICES=4 python3 evaluate_model.py \
#     --model_name google/timesfm-2.0-500m-pytorch \
#     --dataset_names "bike_sharing" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config timesfm_univariate \
#     --folds 1 \
#     --adaptor_method ""

  
# CUDA_VISIBLE_DEVICES=5 python3 evaluate_model.py \
#     --model_name nbeatsx_baseline \
#     --dataset_names "temp" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config nbeatsx_baseline__ \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection "" &

# CUDA_VISIBLE_DEVICES=6 python3 evaluate_model.py \
#     --model_name tide_baseline \
#     --dataset_names "temp" \
#     --batch_size 4 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config tide_baseline \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection "" &


# CUDA_VISIBLE_DEVICES=7 python3 evaluate_model.py \
#     --model_name regression_baseline \
#     --dataset_names "temp" \
#     --batch_size 4 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config KR_baseline_with_cov \
#     --folds 1 \
#     --adaptor_method "" \
#     --features_for_selection "" 

# wait