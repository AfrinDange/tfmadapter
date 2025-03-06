history=(168)  
cuda_device=6
for h in "${history[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_model.py \
        --model_name amazon/chronos-bolt-base \
        --dataset_names "BE" \
        --batch_size 728 \
        --dataset_config dataset_config2 \
        --num_past_k 4 \
        --pos_dims 16 \
        --use_positions \
        --model_config chronos_adaptor_with_cov_no_pseudo_forecaster_h=${h}_ \
        --smallest_history $h \
        --smaller_context $h \
        --folds 1 \
        --adaptor_method "gaussian_process" \
        --remove_pseudo_forecast_generator \
        --features_for_selection "covariates,past_k,positions;covariates,positions;covariates,past_k;covariates" \
        --log_subdir "ablation" &

    ((cuda_device++))  
done

# cuda_device=7
# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_model.py \
#     --model_name amazon/chronos-bolt-base \
#     --dataset_names "FR NP PJM DE BE" \
#     --batch_size 728 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config chronos_adaptor_with_cov_pseudo_forecaster \
#     --smallest_history 168 \
#     --smaller_context 600 \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --features_for_selection "all_covariates,all_past_k,all_positions;all_covariates,all_positions;all_covariates,all_past_k;all_covariates" \
#     --log_subdir "ablation" &
