from typing import List
import jieba
from tqdm import tqdm
import pickle
from gensim import corpora
from gensim.summarization import bm25
import os
import datetime

from file_utils import load_data, save_data

def split_and_rm_stopwords(text:str, stopwords:List[str], lang:str="zh"):
    # split
    if lang == "zh":
        words = jieba.lcut(text)
    elif lang == "en":
        words = text.split(" ")
    else:
        raise NotImplementedError
    # rm stop words
    words_no_stopwords = []
    for w in words:
        if w not in stopwords:
            words_no_stopwords.append(w)
    return words_no_stopwords

def construct_bm25_model_for_pilesky(
    documents_path:str,
    stopwords_path:str,
    output_dir:str,
    lang="zh",
    max_samples=None,
    text_tag:str="text"
):
    print(f"document path: {documents_path}")

    documents = load_data(documents_path)
    if max_samples is not None:
        documents = documents[:max_samples]

    stopwords = [x.strip() for x in open(stopwords_path, "r", encoding="utf-8")]

    # construct corpus, dictionary
    corpus = []
    ids = []
    for doc in tqdm(documents, desc="Construct corpus"):
        passage = doc[text_tag]
        words = split_and_rm_stopwords(passage, stopwords, lang=lang)
        corpus.append(words)
        ids.append(doc["id"])


    dictionary = corpora.Dictionary(corpus)
    end_time = datetime.datetime.now()
    print(f"Len(dictionary) = {len(dictionary)}")
    print(dictionary)

    # # Check dictionary: print 1-st doc's top-5 frequency words
    # doc_vectors = [dictionary.doc2bow(text) for text in corpus]
    # vec1 = doc_vectors[0]
    # vec1_sorted = sorted(vec1, key=lambda x: x[1], reverse=True)
    # print(f"len(vec1_sorted) = {len(vec1_sorted)}")
    # for term, freq in vec1_sorted[:5]:
    #     print(f"{dictionary[term]}: {freq}")
    
    # Create bm25 model
    print(f"\nConstructing bm25Model ...")
    start_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    bm25Model = bm25.BM25(corpus)
    print(f"End time: {end_time}")
    print(f"Spent time = {end_time - start_time}")
    print(f"\nbm25Model is created. corpus_size = {bm25Model.corpus_size}")
    # average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())

    # # Check bm25Model: 
    # query_str = 'Typical generative model approaches include naive Bayes classifier s , Gaussian mixture model s , variational autoencoders and others .'
    # query = []
    # for word in query_str.strip().split():
    #     query.append(word)
    # scores = bm25Model.get_scores(query)
    # # scores.sort(reverse=True)
    # print(f"query_str: {query_str}")
    # print(f"scores: {scores}")

    # Save corpus, dictionary, bm25Model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_model_path = os.path.join(output_dir, "bm25Model.pkl")
    with open(save_model_path, "wb") as f:
        pickle.dump(bm25Model, f)
    
    with open(os.path.join(output_dir, "corpus_of_words.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    dictionary.save(os.path.join(output_dir, "dictionary_of_words.dict"))
    print(f"model saved to: {save_model_path}")


def compute_and_save_bm25_scores(
        model_path:str,
        stopwords_path:str, 
        tar_data_path:str, 
        ref_doc_path:str,
        bm25score_path:str, 
        bm25score_showcase_path:str, 
        lang:str,
        max_samples:int=None, 
        example_tag:str="top_32_similar_examples",
        text_tag_of_tar:str="text",
        text_tag_of_ref:str="text",
        showcase_num:int=10,
        showcase_only:bool=False
    )-> None:
    with open(model_path, "rb") as fb:
        bm25Model = pickle.load(fb)
    print(f"\nLoaded model from: {model_path}")
    
    stopwords = [x.strip() for x in open(stopwords_path, "r", encoding="utf-8")]
    
    print(f"tar_data_path: {tar_data_path}")
    print(f"ref_doc_path: {ref_doc_path}")
    tar_data = load_data(tar_data_path)
    if max_samples is not None:
        tar_data = tar_data[:max_samples]
    ref_doc = load_data(ref_doc_path)
    uniqueId2index = {x[1]["id"]:x[0] for x in enumerate(ref_doc)}
    
    data_w_bm25scores = []
    data_bm25score_showcase = []
    previous_text = ""
    for i_item, item in tqdm(enumerate(tar_data), desc="Get bm25 scores"):
        examp_ids = item[example_tag]
        curr_text = item[text_tag_of_tar]

        curr_query = split_and_rm_stopwords(curr_text, stopwords, lang=lang)

        curr_scores = []
        curr_score_showcase = []
        for tmp_id in examp_ids:
            tmp_index = uniqueId2index[tmp_id]
            tmp_score = bm25Model.get_score(curr_query, tmp_index)
            curr_scores.append(tmp_score)

            if len(data_bm25score_showcase) < showcase_num:
                if curr_text == previous_text:
                    continue
                curr_score_showcase.append({
                    "text": ref_doc[tmp_index][text_tag_of_ref],
                    "score": tmp_score
                })

        data_w_bm25scores.append({
            "id": item["id"],
            "text": item[text_tag_of_tar],
            example_tag: curr_scores
        })

        if len(data_bm25score_showcase) < showcase_num:
            if curr_text == previous_text:
                continue
            data_bm25score_showcase.append({
                "id": item["id"],
                "text": item[text_tag_of_tar],
                "doc_score": curr_score_showcase
            })
        
        previous_text = curr_text

    bm25score_dir = os.path.dirname(bm25score_path)
    if not os.path.exists(bm25score_dir):
        os.makedirs(bm25score_dir)
    save_data(bm25score_path, data_w_bm25scores)
    save_data(bm25score_showcase_path, data_bm25score_showcase)
    print(f"bm25score_path: {bm25score_path}")
    print(f"bm25score_showcase_path: {bm25score_showcase_path}")

if __name__ == "__main__":
    # Sky-NER
    # Construct bm25 model
    lang = "zh"
    for subset in ["5k_random_42", "5k_random_67", "10k_random_42", "10k_random_67", "total"]:
        construct_bm25_model_for_pilesky(
            documents_path=f"universal-ner-zh/sky_gpt3.5_{subset}/train_allinone.jsonl",
            stopwords_path=f"Process-data/src/resources/my_combined_stopwords.txt",
            output_dir=f"universal-ner-zh/sky_gpt3.5_{subset}/bm25Model",
            lang=lang,
        )

    # Compute and save bm25 scores
    showcase_num=100
    for subset in ["5k_random_42", "5k_random_67", "10k_random_42", "10k_random_67", "total"]:
        compute_and_save_bm25_scores(
            model_path=f"universal-ner-zh/sky_gpt3.5_{subset}/bm25Model/bm25Model.pkl",
            stopwords_path=f"Process-data/src/resources/my_combined_stopwords.txt", 
            tar_data_path=f"universal-ner-zh/sky_gpt3.5_{subset}/train_allinone_similar_example_ids_128_GTElargeEmb_text.jsonl",
            ref_doc_path=f"universal-ner-zh/sky_gpt3.5_{subset}/train_allinone.jsonl",
            bm25score_path=f"universal-ner-zh/sky_gpt3.5_{subset}/bm25Score/train_simexamp_bmscores_128_GTElargeEmb_text.jsonl",
            bm25score_showcase_path=f"universal-ner-zh/sky_gpt3.5_{subset}/bm25Score/train_simexamp_bmscores_showcase-{showcase_num}_128_GTElargeEmb_text.json", 
            lang=lang,
            example_tag="top_128_similar_examples",
            showcase_num=showcase_num
        )


    # Generate BM25 scores for Chinese benchmarks
    dataname = "boson"
    
    setname="5k_random_42"
    outdomain_data_dir=f"universal-ner-zh/sky_gpt3.5_{setname}"
    outdomain_dataname="piletype"
    
    bench_data_dir=f"my_benchs/zh_sampled/{dataname}"
    bench_data_rag_dir=f"my_benchs/zh_sampled_rag/{dataname}"
    showcase_num=50
    lang="zh"
    compute_and_save_bm25_scores(
        model_path=f"{outdomain_data_dir}/bm25Model/bm25Model.pkl",
        stopwords_path=f"Process-data/src/resources/my_combined_stopwords.txt",
        tar_data_path=f"{bench_data_rag_dir}/sim_examp_ids_from_outdomain/test_simexample_ids_128_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
        ref_doc_path=f"{outdomain_data_dir}/train_allinone.jsonl",
        bm25score_path=f"{bench_data_rag_dir}/bm25Score_outdomain/test_simexample_bmscores_128_{outdomain_dataname}_{setname}_GTElargeEmb_text.jsonl",
        bm25score_showcase_path=f"{bench_data_rag_dir}/bm25Score_outdomain/showcase-{showcase_num}_test_simexamp_bmscores_128_{outdomain_dataname}_{setname}_GTElargeEmb_text.json", 
        lang=lang,
        example_tag="top_128_similar_examples",
        text_tag_of_tar="sentence",
        showcase_num=showcase_num
    )

