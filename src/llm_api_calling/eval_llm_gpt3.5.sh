# Evaluating LLM on benchmarks
datanames=("ontonotes4zh" "msra")
model=gpt-3.5-turbo-0125
max_len=1024

prompt_config_path=configs/prompt_config/ner_llm_evaluation.json
prompt_version=prompt_v0_json

for dataname in ${datanames[@]}
do

input_path=data/benchmark_data/regular/${dataname}/test.jsonl
label_info_path=data/benchmark_data/regular/${dataname}/abb2labelname.json
output_dir=outputs/llm_api_calling/llm_evaluation/${model}/${prompt_version}/${dataname}

python src/llm_api_calling/AskLLM.py \
    --dataname $dataname \
    --input_path $input_path \
    --output_dir $output_dir \
    --label_info_path $label_info_path \
    --model $model \
    --max_len $max_len \
    --prompt_config_path $prompt_config_path \
    --prompt_version $prompt_version \
    --llm_using llm_evaluation \
    --do_asking \
    --resume \
    --do_parsing \
    --do_compute_metric \

done