cuda_device=0


CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name Salesforce/moirai-1.1-R-base --dataset_names "covid-national  electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --pos_dims 0 --model_config moirai_no_cov

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name Salesforce/moirai-1.1-R-base --dataset_names "covid-national  electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --pos_dims 0 --model_config moirai_with_cov

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name Salesforce/moirai-1.1-R-base --dataset_names "covid-national electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 4 --model_config nw_kernel_with_cov_pastk0_pos4

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name Salesforce/moirai-1.1-R-base --dataset_names "covid-national electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 8 --model_config gp_with_cov_pastk4_pos8_width2.0_variance2.4

CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name Salesforce/moirai-1.1-R-base --dataset_names "covid-national electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 8 --model_config gp_variance_adjust_with_cov_pastk4_pos8_width2.0_variance2.4


# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "covid-national  electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --pos_dims 0 --model_config chronos_univariate

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "covid-national" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 4 --model_config nw_kernel_with_cov_h672_pastk0_pos4

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "covid-national electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 8 --model_config gp_with_cov_h672_pastk4_pos8_width2.0_variance2.4

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "covid-national electric_consumption" --batch_size 128 --dataset_config dataset_config2 --num_past_k 0 --use_positions --pos_dims 8 --model_config gp_variance_adjust_with_cov_h672_pastk4_pos8_width2.0_variance2.4

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "BE DE NP PJM FR" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 4 --use_positions --pos_dims 16 --model_config gp_variance_adjust_with_cov_h672_pastk4_pos16_width2.0_variance2.4

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_chronos.py --model_name amazon/chronos-bolt-base --dataset_names "BE DE NP PJM FR" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 8 --use_positions --pos_dims 16 --model_config nw_kernel_with_cov_h672_pastk8_pos16

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE DE NP PJM FR" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 168 --rolling_window_size 24 --prediction_length 24 --num_past_k 4 --use_positions --pos_dims 16 --patch_size 32 --model_config gp_with_cov_h672_p32_pastk4_pos16_kernel_w2.0_v2.4 


# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 0 --rolling_window_size 0 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 32 --model_config moirai_with_cov_h672_p32

# CUDA_VISIBLE_DEVICES=$cuda_device python3 evaluate_moirai.py --model_name "Salesforce/moirai-1.1-R-base" --dataset_names "BE FR DE NP PJM" --batch_size 128 --dataset_config dataset_config2 --context_length 672 --smaller_context 0 --rolling_window_size 0 --prediction_length 24 --num_past_k 0 --pos_dims 0 --patch_size 32 --model_config moirai_no_cov_h672_p32