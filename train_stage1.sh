export MODEL_DIR="runwayml/stable-diffusion-v1-5" # your SD path
export OUTPUT_DIR="stage1"  # your save path
export CONFIG="./default_config.yaml"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $CONFIG train_stage1.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --source_column="target" \
    --target_column="source" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="no" \
    --train_data_dir "data.jsonl" \  # your data.jsonl path
    --resolution=512 \ 
    --learning_rate=5e-5 \
    --train_batch_size=16 \
    --num_validation_images=2 \
    --validation_ids "1.png" "2.png" \  # your validation image paths
    --gradient_accumulation_steps=1 \
    --num_train_epochs=500 \
    --validation_steps=2000 \
    --checkpointing_steps=2000
