export MODEL_NAME = "CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR = "./dreambooth-ox/images"
export OUTPUT_DIR = "stable-diffusion-oxified"
export INSTANCE_PROMPT = "an image of the oxenai ox"

accelerate launch diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --hub_model_id="YOUR-HF-NAMESPACE/YOUR-MODEL-NAME" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub