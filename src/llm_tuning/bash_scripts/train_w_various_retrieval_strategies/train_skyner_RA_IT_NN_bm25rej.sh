cuda_id=0
# model
model_path=models/Qwen1.5-7B-Chat
modelname=qwen1.5-7b
template=qwen
# training
stage=sft
finetuning_type=lora
lora_target=all
epoch=3.0

subset=5k_random_42
dataset_dir=data/it_data/sky_ner_${subset}
dataname=train_w_NN_bm25Rej_2_128_20 # RA-IT with NN and bm25-filtering
exp_name=skyner_${subset}_RA_IT_NN_bm25rej

weight_save_path=weights/${modelname}/${finetuning_type}/${stage}/${exp_name}

echo -e "weight_save_path: $weight_save_path"
echo -e "model_path: $model_path\n"

# ---- do train ----
CUDA_VISIBLE_DEVICES=${cuda_id} python src/llm_tuning/LLaMA-Factory/train.py \
    --stage ${stage} \
    --do_train \
    --model_name_or_path ${model_path} \
    --dataset_dir ${dataset_dir} \
    --dataset ${dataname} \
    --template ${template} \
    --finetuning_type ${finetuning_type} \
    --lora_target ${lora_target} \
    --output_dir ${weight_save_path} \
    --cutoff_len 2048 \
    --per_device_train_batch_size 8 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --warmup_ratio 0.03 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs ${epoch} \
    --val_size 0.1 \
    --plot_loss \
    --fp16

# ---- do merging lora ----
weight_save_path=${weight_save_path}_lora_merged
CUDA_VISIBLE_DEVICES=${cuda_id} python src/llm_tuning/LLaMA-Factory/export_model.py \
    --model_name_or_path $model_path \
    --adapter_name_or_path $weight_save_path \
    --template $template_fac \
    --finetuning_type $finetuning_type \
    --export_dir $lora_merged_path \
    --export_size 2 \
    --export_legacy_format False
