model_names=("google/timesfm-2.0-500m-pytorch") # "amazon/chronos-bolt-base" "Salesforce/moirai-1.1-R-base" 
model_shortnames=("timesfm") # "chronos" "moirai" 
# devices=(2 7 2 7 7)
devices=(0 2 0 7 7)
for covariate in 1 2 4; do
    for i in "${!model_names[@]}"; do
        model_name="${model_names[$i]}"
        model="${model_shortnames[$i]}"
        device="${devices[$covariate]}"

        echo "Running $model_name with use_covariates=$covariate on CUDA:$device"

        CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
            --model_name "$model_name" \
            --dataset_names "temp bafu electricity air_quality climate meteo drift chlorine" \
            --batch_size 1024 \
            --dataset_config dataset_config2 \
            --num_past_k 4 \
            --pos_dims 16 \
            --use_positions \
            --model_config "${model}_adaptor_with_cov_${covariate}" \
            --folds 1 \
            --adaptor_method "gaussian_process" \
            --log_subdir "varying_covariates" \
            --use_covariates $covariate &
    done
done
wait

echo "All evaluations completed."

# covariate=1
# CUDA_VISIBLE_DEVICES=2 python3 evaluate_model.py \
#     --model_name "google/timesfm-2.0-500m-pytorch" \
#     --dataset_names "air_quality" \
#     --batch_size 1024 \
#     --dataset_config dataset_config2 \
#     --num_past_k 4 \
#     --pos_dims 16 \
#     --use_positions \
#     --model_config "timesfm_adaptor_with_cov_${covariate}" \
#     --folds 1 \
#     --adaptor_method "gaussian_process" \
#     --log_subdir "varying_covariates" \
#     --use_covariates $covariate \
#     --test_run