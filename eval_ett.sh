models=("small" "base" "large")
devices=(2 6 7)

for i in "${!models[@]}"; do
    variant="${models[$i]}"
    device="${devices[$i]}"

    model_name="Salesforce/moirai-1.1-R-${variant}"
    model="moirai-${variant}"

    echo "Running model: $model_name on CUDA device: $device"

    CUDA_VISIBLE_DEVICES=$device python3 evaluate_model.py \
        --model_name "$model_name" \
        --dataset_names "ETTh1 ETTh2 ETTm1 ETTm2" \
        --batch_size 1024 \
        --dataset_config dataset_config2 \
        --num_past_k 4 \
        --pos_dims 16 \
        --use_positions \
        --model_config "${model}_adapter_with_cov" \
        --folds 1 \
        --adaptor_method "gaussian_process" \
        --log_subdir "moirai_ablation_2.0__" &
done

wait
