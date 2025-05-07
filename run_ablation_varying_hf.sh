# model_name="amazon/chronos-bolt-base"
# model="chronos"

# model_name="Salesforce/moirai-1.1-R-base"
# model="moirai"

model_name="google/timesfm-2.0-500m-pytorch"
model="timesfm"

device=5

for context_length in 672 1344 2016; do
    if [ "$context_length" -lt 168 ]; then
        smallest_history=24
    else
        smallest_history=168
    fi

    smaller_context=$((context_length - 72))

    CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
        --model_name "$model_name" \
        --dataset_names "NP BE DE FR PJM" \
        --batch_size 1024 \
        --dataset_config dataset_config2 \
        --num_past_k 4 \
        --pos_dims 16 \
        --use_positions \
        --model_config "${model}_univariate_H${context_length}_h${smallest_history}_f24" \
        --folds 1 \
        --adaptor_method "gaussian_process" \
        --log_subdir "rebuttal-ablation-vary" \
        --context_length "$context_length" \
        --smaller_context "$smaller_context" \
        --smallest_history "$smallest_history"
done

for prediction_length in 168; do

    windows=$(( 728 / (prediction_length / 24) ))
    smaller_context=$((672 - 3 * prediction_length))

    CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
        --model_name "$model_name" \
        --dataset_names "NP BE DE PJM" \
        --batch_size 1024 \
        --dataset_config dataset_config2 \
        --num_past_k 4 \
        --pos_dims 16 \
        --use_positions \
        --model_config "${model}_adaptor_with_cov_f${prediction_length}" \
        --folds 1 \
        --adaptor_method "gaussian_process" \
        --log_subdir "rebuttal-ablation-vary" \
        --prediction_length "$prediction_length" \
        --rolling_window_size "$prediction_length" \
        --distance "$prediction_length" \
        --smaller_context "$smaller_context" \
        --windows "$windows" 

    windows=$(( 364 / (prediction_length / 24) ))

    CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
        --model_name "$model_name" \
        --dataset_names "FR" \
        --batch_size 1024 \
        --dataset_config dataset_config2 \
        --num_past_k 4 \
        --pos_dims 16 \
        --use_positions \
        --model_config "${model}_adaptor_with_cov_f${prediction_length}" \
        --folds 1 \
        --adaptor_method "gaussian_process" \
        --log_subdir "rebuttal-ablation-vary" \
        --prediction_length "$prediction_length" \
        --rolling_window_size "$prediction_length" \
        --distance "$prediction_length" \
        --smaller_context "$smaller_context" \
        --windows "$windows" 
done

