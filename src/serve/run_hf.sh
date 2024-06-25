# English
CUDA_VISIBLE_DEVICES=4 python src/serve/hf.py \
    --model_path models/RA-IT-NER \
    --max_new_tokens 256 \
    --language en

# Chinese
# CUDA_VISIBLE_DEVICES=4 python src/serve/hf.py \
#     --model_path models/RA-IT-NER-zh \
#     --max_new_tokens 256 \
#     --language zh