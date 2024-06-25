# Use LLM to generate NER outputs for constructing IT dataset
model=gpt-3.5-turbo-0125
max_len=1024

prompt_config_path=configs/prompt_config/ner_llm_annotation.json
prompt_version=prompt_v0_json

input_path=data/corpus_data/my_chunk_256_data_sampled/sky_10_samples.chunk_256_sampled.jsonl
dataname=sky_samples
output_dir=outputs/llm_api_calling/llm_annotation/${model}/${prompt_version}/${dataname}

python src/llm_api_calling/AskLLM.py \
    --input_path $input_path \
    --output_dir $output_dir \
    --lang zh \
    --model $model \
    --max_len $max_len \
    --prompt_config_path $prompt_config_path \
    --prompt_version $prompt_version \
    --llm_using llm_annotation \
    --resume \
    --do_asking \
    --do_parsing \
    --max_samples 20
    