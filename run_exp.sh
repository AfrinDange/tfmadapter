
model_names=(
    "Salesforce/moirai-1.1-R-small"
    "Salesforce/moirai-1.1-R-base"
    "Salesforce/moirai-1.1-R-large"
    "Salesforce/moirai-1.0-R-small"
    "Salesforce/moirai-1.0-R-base"
    "Salesforce/moirai-1.0-R-large"
    "Salesforce/moirai-moe-1.0-R-small"
    "Salesforce/moirai-moe-1.0-R-base"
)

# for model_name in "${model_names[@]}"
# do
#     CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py \
#         --dataset_names "BE FR DE NP PJM" \
#         --model_config "${model_name}_no_cov" \
#         --patch_size 32 \
#         --batch_size 512 \
#         --dataset_config dataset_config2 \
#         --model_name "$model_name" &
# done

# kernel regression 
# with covariates => (increased history, past_k, fixed_history, positions, patch size)

cuda_device=5

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --prediction_length 24 --patch_size 8 --model_config moirai_no_cov_h672_p8 

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --prediction_length 24 --patch_size 32 --model_config moirai_no_cov_h672_p32 

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --prediction_length 24 --patch_size 8 --model_config moirai_with_cov_h672_p8 

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --prediction_length 24 --patch_size 32 --model_config moirai_with_cov_h672_p32 

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 168 --smaller_context 144 --rolling_window_size 24 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 8 --model_config kernel_regr_with_cov_h168_p8

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 168 --smaller_context 144 --rolling_window_size 24 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 8 --model_config kernel_regr_with_cov_h168_p8


CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 168 --smaller_context 144 --rolling_window_size 24 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 32 --model_config kernel_regr_with_cov_h168_p32


CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 8 --model_config kernel_regr_with_cov_h672_p8


CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 32 --model_config kernel_regr_with_cov_h672_p8

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 4 --pos_dims 0 --patch_size 32 --model_config kernel_regr_with_cov_h672_p8_pastk4

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 8 --pos_dims 0 --patch_size 32 --model_config kernel_regr_with_cov_h672_p8_pastk8

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 16 --pos_dims 0 --patch_size 32 --model_config kernel_regr_with_cov_h672_p8_pastk16


# CUDA_VISIBLE_DEVICES=6 python3 evaluate_moirai.py --dataset_names "BE FR DE NP PJM" --model_config moirai_with_cov_p_8 --patch_size 8 --batch_size 512 --dataset_config dataset_config2 
# CUDA_VISIBLE_DEVICES=6 python3 evaluate_moirai.py --dataset_names "covid-national" --model_config moirai_no_cov --patch_size 32 --batch_size 512 --dataset_config dataset_config2  --model_name Salesforce/moirai-1.1-R-small &
# CUDA_VISIBLE_DEVICES=4 python3 evaluate_moirai.py --dataset_names "BE" --model_config moirai_with_cov --patch_size 32 --batch_size 128 --dataset_config dataset_config2 --test_run --model_name "Salesforce/moirai-1.1-R-small" &
# CUDA_VISIBLE_DEVICES=5 python3 evaluate_moirai.py --dataset_names "BE" --model_config moirai_no_cov --patch_size 32 --batch_size 128 --dataset_config dataset_config2 --test_run --model_name "Salesforce/moirai-1.1-R-small" 

# CUDA_VISIBLE_DEVICES=3 python3 evaluate_moirai.py --dataset_names "BE" --model_config kernel_extended_with_cov_1.0 --patch_size 32 --batch_size 1024 --dataset_config dataset_config2 & 
# CUDA_VISIBLE_DEVICES=3 python3 evaluate_moirai.py --dataset_names "FR" --model_config kernel_extended_with_cov_1.0 --patch_size 32 --batch_size 1024 --dataset_config dataset_config2 & 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "PJM" --model_config kernel_extended_with_cov_1.0 --patch_size 32 --batch_size 1024 --dataset_config dataset_config2 & 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "NP" --model_config kernel_extended_with_cov_1.0 --patch_size 32 --batch_size 1024 --dataset_config dataset_config2 & 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "DE" --model_config kernel_extended_with_cov_1.0 --patch_size 32 --batch_size 1024 --dataset_config dataset_config2 & 
wait 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "BE FR DE NP PJM" --model_config kernel_regr_with_cov_3f_poly3_p_8 --patch_size 8 --batch_size 512 --dataset_config dataset_config2 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "BE FR DE NP PJM" --model_config ensemble_with_cov_rbf_poly5_linear --patch_size 32 --batch_size 512 --dataset_config dataset_config2 
# CUDA_VISIBLE_DEVICES=7 python3 evaluate_moirai.py --dataset_names "BE FR DE NP PJM" --model_config ensemble_with_cov_3f_poly3_p_8 --patch_size 8 --batch_size 512 --dataset_config dataset_config2
