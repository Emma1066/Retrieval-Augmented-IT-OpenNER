cuda_id=0
# model
modelname=qwen1.5-7b
template=qwen

# training
stage=sft
finetuning_type=lora

exp_name=skyner_${subset}_RA_IT_MixedNN

model_path=weights/${modelname}/${finetuning_type}/${stage}/${exp_name}_lora_merged

# ---- do eval (vllm) ----
temperature=0
top_p=1.0
max_tokens=1024

data_format=conversation_sharegpt

dataname=boson
# no example
data_path=data/benchmark_it_data/${dataname}/test.json
category=zero-shot
output_dir=outputs/${modelname}/${finetuning_type}/${stage}/${exp_name}/${category}/${dataname}/

CUDA_VISIBLE_DEVICES=${cuda_id} python src/llm_tuning/evaluation/evaluate.py \
    --model_path ${model_path} \
    --conversation_template $template \
    --data_path ${data_path} \
    --data_format ${data_format} \
    --output_dir ${output_dir} \
    --temperature ${temperature} --top_p ${top_p} --max_tokens ${max_tokens} \
    --gpu_memory_utilization 0.9

# out-domain example, NN
data_path=data/benchmark_it_data_rag/${dataname}/test_w_NN_2_skygpt3.5_5k_random_42_GTElargeEmb_text.json
category=outdomain_NN
output_dir=outputs/${modelname}/${finetuning_type}/${stage}/${exp_name}/${category}/${dataname}/

CUDA_VISIBLE_DEVICES=${cuda_id} python src/llm_tuning/evaluation/evaluate.py \
    --model_path ${model_path} \
    --conversation_template $template \
    --data_path ${data_path} \
    --data_format ${data_format} \
    --output_dir ${output_dir} \
    --temperature ${temperature} --top_p ${top_p} --max_tokens ${max_tokens} \
    --gpu_memory_utilization 0.9
