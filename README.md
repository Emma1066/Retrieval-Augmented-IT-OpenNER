# RA-IT-NER

This is the github repository for the paper: *Retrieval Augmented Instruction Tuning for Open NER with Large Language Models*.

- 📖 Paper: [Retrieval Augmented Instruction Tuning for Open NER with Large Language Models](https://todo)
- 📙 Data: [Sky-NER](https://TODO), Our constructed instruction tuning data for Chinese open NER. 
- 🔮 Models: [RA-IT-NER](https://TODO) and [RA-IT-NER-zh](https://TODO) , models trained with our retrieval augmented instruction tuning (RA-IT) approach.

## Introduction

<img style="width:35%;" align="right" src=assets/method.png/>

* This work explores Retrieval Augmented Instruction Tuning (RA-IT) for open NER. For each training sample, we retrieve semantically similar examples from the training dataset as the context and prepend them to the input of the original instruction. 
* We construct a Chinese IT dataset for open NER and evaluate RA-IT in both English and Chinese scenarios. Experimental results verify the effectiveness of RA-IT across various data sizes and in both English and Chinese scenarios.
* we suggest implementing on-demand RAG for inference after RA-IT. When sufficient in-domain examples are available, conduct RAG with similar examples to boost inference. When only out-domain examples are available, apply an example filtering method such as BM25 scoring for RAG, or simply conduct inference without examples.



## Installation
We run this repository based on the following dependencies:
```bash
python==3.11.5
pytorch==2.3.0
transformers=4.41.2
peft==0.11.1
openai==1.21.2
flash_attn==2.5.9
vllm==0.4.3
```
You will also need these dependencies:
```bash
numpy tqdm rich datasets Jinja jieba pandas pyarrow
```


## Data
### Chinese OpenNER Data Construction
We release [Sky-NER](https://todo), an instruction-tuning dataset constructed for Chinese openNER, based on the Sky corpus. We followed the recipe in [UniversalNER](https://arxiv.org/abs/2308.03279) to construct Sky-NER.

We also release the code of our data construction pipeline in [data_process](src/data_process/).


### Training and Evaluation Datasets
We provide processed **benchmark datasets** used in our paper at the [Google Drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing). We also include our **training datasets** in this package, including the sampled 5K and 10K datasets used in our paper.

You can download and unzip the data pacakge and put the content in the [data](data) folder.

The code for **generating RA-IT data** and **preprocessing the benchmarks** can all be found in [data_process](src/data_process/).


## Models
We release our models fine-tuned with the proposed RA-IT approach, [RA-IT-NER](todo) and [RA-IT-NER-zh](todo), which are trained on the English NER dataset [Pile-NER](todo) and the Chinese NER dataset [Sky-NER](todo) respectively.

| Model           | Language | Backbone    | Link |
|-----------------|----------|-------------|------|
| RA-IT-NER-8B    | English  | Llama-3-8B  | TODO |
| RA-IT-NER-zh-7B | Chinese  | Qwen-1.5-7B | TODO |

## Demo
The inferece code are based on [vllm](https://github.com/vllm-project/vllm).

Please download our fine-tuned models [RA-IT-NER](todo) and [RA-IT-NER-zh](todo) and put them in the model folders before running the demos.

The following commands for running demos can be found in the bash scripts in [serve](src/serve).

Our model RA-IT-NER supports inference with and without RAG. 

### Gradio Web UI

Use the following command to launch a Gradio demo locally:
```Shell
python src/serve/gradio_server.py \
    --model_path models/RA-IT-NER \
    --tensor_parallel_size 1 \
    --max_input_length 2048 \
    --language en
```

### CLI Inference

Use the following command to do inference with vllm:
```Shell
python src/serve/cli.py \
    --model_path models/RA-IT-NER \
    --tensor_parallel_size 1 \
    --max_input_length 2048 \
    --language en
```

Use the following command to do inference with HuggingFace Transformers:
```Shell
python src/serve/hf.py \
    --model_path models/RA-IT-NER \
    --max_new_tokens 256 \
    --language en
```

## Finetuning

We use [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune our models. 

Generate the RA-IT datasets using code in [here](src/data_process), or download the processed RA-IT training data from [google drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing). Download the base model from huggingface.

Run the bash scripts for finetuning:
```bash
# Training with RA-IT
sh src/llm_tuning/bash_scripts/train_skyner_RA_IT.sh
# Training with Vanilla IT
sh src/llm_tuning/bash_scripts/train_skyner_vanilla_IT.sh
```

We provide the scripts of training with various retrieval strategies in [here](src/llm_tuning/bash_scripts):


## Evaluation

Download the processed benchmark data from [google drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing). Or process new benchmarks with the code in [data_process](src/data_process/).

Our evaluation code is adapted from [UniversalNER](https://github.com/universal-ner/universal-ner/tree/main). 

Run the bash scripts for evaluating:
```bash
# Evaluation of RA-IT model
sh src/llm_tuning/bash_scripts/eval_skyner_RA_IT.sh
# Evaluation of Vanilla IT model
sh src/llm_tuning/bash_scripts/eval_skyner_vanilla_IT.sh
```
For inference with various retrieval strategies, see more commands in the script [eval_skyner_RA_IT.sh](src/llm_tuning/bash_scripts/eval_skyner_RA_IT.sh). Uncomment the commands of the retrieval strategies you'd like to evaluate and then run the script.

## Acknowledgement
This repository is built based upon the excellent work of [UniversalNER](https://github.com/universal-ner/universal-ner/tree/main) and [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory). Also, the data preprocessing also partially referenced [MINI_LLM](https://github.com/jiahe7ay/MINI_LLM). We thank them for their open-source contributions.
