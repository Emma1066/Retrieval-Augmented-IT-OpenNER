# Data Processing
## Chinese OpenNER Data Construction
We release [Sky-NER](https://todo), an instruction-tuning dataset constructed for Chinese openNER, based on the Sky corpus. We followed the recipe in [UniversalNER](https://arxiv.org/abs/2308.03279) to construct Sky-NER. You can also download the datasets of Sky-NER in [google drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing).

We also release the code of our data construction pipeline here.
All the generated data files will be saved in [corpus_data](../../data/corpus_data/), and we have put a small number of example data in this folder for quick exploration. We also put the ChatGPT generated outputs of these examples in the folder [output](outputs/llm_api_calling/llm_annotation/gpt-3.5-turbo-0125/prompt_v0_json/sky_10_samples/entity_statistics.json) for reference.

1. Preprocess the corpus data using [1.preprocess_corpus_data.py](1.preprocess_corpus_data.py).
2. Chunk the corpus passages into a maximum length of 256 tokens using [2.chunk_corpus.py](2.chunk_corpus.py).
3. Generate NER outputs with ChatGPT by runing the scripts [ask_llm_gpt3.5_to_annotate_sky.sh](../llm_api_calling/ask_llm_gpt3.5_to_annotate_sky.sh).
```shell
# Generate NER outputs with ChatGPT
sh src/llm_api_calling/ask_llm_gpt3.5_to_annotate_sky.sh
```
1. Filtering the NER outputs generated by ChatGPT using [3.ner_annotation_filter.py](3.ner_annotation_filter.py).
2. Generate conversation style instruction tuning data using [4.gen_it_data.py](4.gen_it_data.py).

We write all codes of calling the processing functions in the main function of each of the above python file. Uncomment the part you need and then run the python file.


## RA-IT data construction
The generated RA-IT datasets can be downloaded from [google drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing) for directly using.

The steps for generating RA-IT training data is as below. All generated data files will be saved in [it_data](../../data/it_data/).

1. Generate embeddings using [gen_emb.py](gen_emb.py).
2. The example code of retrieving examples for training data is summarized in the function `retrieve_example_for_training_data` in [5.gen_rag_it_data.py](5.gen_rag_it_data.py). The pipeline is as follows:
   1. Call `retrieve_similar_examples` function to retrieve semantically similar examples, i.e., nearest neighbor (NN) examples, for each training sample. This generates the data file that save the example ids for each training sample.
   2. Call `add_examp_to_multi_turn_conv_data` function to add the examples into the original vanilla conversation-style instruction. This generates the final RA-IT data files.
   3. Generate bm25 scores using [gen_bm25.py](gen_bm25.py). Then call `add_adaptive_examp_to_multi_turn_conv_data` function to use various retrieval strategies and example filtering methods, e.g. diverse NN and bm25 scoring.

## Evaluation data construction
We convert the format of benchmark data for evaluation. The generated evaluation datasets can be downloaded from [google drive](https://drive.google.com/file/d/1lJZd89KwfIaIQKfty7Ba1nvkhhUKqPjz/view?usp=sharing) for directly using.

Before introducing the steps for generating RA-IT training data, we first introduce the **folder structure** of saving data files generated during different steps. The verbalized benchmark datasets and other generated data files are saved in different subfolders in folder [data](../../data) according to their usage. The introduction of each subfolder as follows:
* `benchmark_data`: The verbalized benchmark datasets, including original data splits and our sampled data sets.
* `benchmark_data_rag`: The intermediate files used for generating RAG data. The RAG data will be used for inference with RAG.
* `benchmark_it_data`: The instruction format of benchmark data, which are used for final evaluation.
* `benchmark_it_data`: The instruction format of benchmark data with RAG, which are used for final evaluation. This is generated with the help of the intermediate files saved in folder `benchmark_data_rag`.

The steps for generating evaluation datasets are as below.

1. Generate embeddings using [gen_emb.py](gen_emb.py).
2. The example code of retrieving examples for evaluation data is summarized in the function `retrieve_outdomain_example_for_benchmarks` and `retrieve_indomain_example_for_benchmarks` in [5.gen_rag_it_data.py](5.gen_rag_it_data.py). We use take the code in `retrieve_outdomain_example_for_benchmarks` as the example to introduce the main pipeline:
   1. Call `retrieve_similar_examples` function to retrieve semantically similar examples, i.e., nearest neighbor (NN) examples, for each evaluation sample. This generates the data file that save the example ids for each evaluation sample.
   2. Call `benchmark_test_set_to_rag_conversation` function to generate conversation-style evaluation data with the retrieved examples prepended.
   3. Generate bm25 scores using [gen_bm25.py](gen_bm25.py). Then call `benchmark_test_set_to_adaptive_rag_conversation` function to use various retrieval strategies and example filtering methods, e.g. diverse NN and bm25 scoring.
   4. Call `retrieve_random_examples` function to retrieve random examples. Then call `benchmark_test_set_to_rag_conversation` to generate conversation-style evaluation data with the retrieved examples prepended.