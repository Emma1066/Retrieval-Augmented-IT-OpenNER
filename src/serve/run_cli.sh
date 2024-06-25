# English
CUDA_VISIBLE_DEVICES=4 python src/serve/cli.py \
    --model_path models/RA-IT-NER \
    --tensor_parallel_size 1 \
    --max_input_length 2048 \
    --language en

# Chinese
# CUDA_VISIBLE_DEVICES=4 python src/serve/cli.py \
#     --model_path models/RA-IT-NER-zh \
#     --tensor_parallel_size 1 \
#     --max_input_length 2048 \
#     --language zh