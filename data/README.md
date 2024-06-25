These are the datasets used in our paper "Retrieval Augmented Instruction Tuning for Open NER with Large Language Models", including the instruction-tuning training datasets and evaluation benchmarks.

## Download
Please download the training and evaluation datasets from [Google Drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing). Unzip the data package, then put them in this data folder.

## Directory
* `benchmark_data`: The processed benchmark datasets, including original data splits and our sampled data sets.
* `benchmark_data_rag`: The intermediate files used for generating RAG data. The RAG data will be used for inference with RAG.
* `benchmark_it_data`: The instruction format of benchmark data, which are used for final evaluation.
* `benchmark_it_data`: The instruction format of benchmark data with RAG, which are used for final evaluation. This is generated with the help of the intermediate files saved in folder `benchmark_data_rag`.
* `it_data`: The instruction-tuning data used to train the models, including Sky-NER and Pile-NER, and also our sampled dataset of sizes 5K and 10K.

