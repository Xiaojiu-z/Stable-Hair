export MODEL_DIR="runwayml/stable-diffusion-v1-5" # your SD path
export OUTPUT_DIR="stage2"  # your save path
export CONFIG="./default_config.yaml"

CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch --config_file $CONFIG train_stage2.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --refer_column="reference" \
    --source_column="source" \
    --target_column="target" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="no" \
    --train_data_dir "your_data_jsonl_path.jsonl" \
    --resolution=512 \
    --learning_rate=2e-5 \
    --train_batch_size=6 \
    --num_validation_images=2 \
    --validation_ids "1.jpg" "2.jpg" \
    --validation_hairs "1.jpg" "2.jpg" \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=100 \
    --validation_steps=5000 \
    --checkpointing_steps=5000