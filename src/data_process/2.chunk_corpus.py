from langchain.text_splitter import RecursiveCharacterTextSplitter
from rich import progress
import os
import fastparquet
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import ujson
import random

CHUNK_LEN = 256

def delete_file(file: str)-> bool:
    '''
    询问删除文件
    '''
    if os.path.exists(file):
        ans = input('delete file: {} ? Yes (y) or No (n)'.format(file))
        ans = ans.lower()
        if ans in ('yes', 'y'):
            os.remove(file)
            print('deleted.')
            return True
    return False

def parquet_to_jsonl(
        parquet_file:str = 'Chinese-corpus/my_data/wiki_cn_filtered.parquet',
        json_file:str = 'Chinese-corpus/my_data/wiki_cn_filtered.jsonl'
) -> None:
    '''
    将parquet文件转换为json
    '''
    if os.path.exists(json_file):
        assert delete_file(json_file)

    source_pf = fastparquet.ParquetFile(parquet_file)
    cur_rows = []
   
    for pf_chunk in progress.track(source_pf):
        for rows in pf_chunk.iter_row_groups():
            for txt in rows['text']:
                if len(txt) == 0: continue
                cur_rows.append({"text":txt})

    with open(json_file, 'w', encoding='utf-8') as f:
        for item in progress.track(cur_rows):
            f.write(ujson.dumps(item, ensure_ascii=False)+"\n")

def write_jsonl_file(file: str, data: list):
    with open(file, "w", encoding="utf-8") as wf:
        for item in data:
            wf.write(ujson.dumps(item, ensure_ascii=False) + "\n")

def recursive_char_text_split(data, chunk_size:int=256, chunk_overlap:int=0):
    separators = ["。", "！", "？", "\n\n", "\n", "；", " "]

    print(f"Load tokenizer from tiktoken encoder...")
    r_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True
    )

    chunked_data = []
    for item in progress.track(data, description="chunking", total=len(data)):
        chunked_txt = r_splitter.split_text(item["text"])
        chunked_data.extend(chunked_txt)
    
    print(f"len_original_len = {len(data)}")
    print(f"len_chunked_data = {len(chunked_data)}")
    return chunked_data

def batch_move_ending_punc(data: list[str]) -> list[str]:
    puncs = ["。", "！", "？", "；"]
    for i in range(1,len(data)):
        start_char = data[i][0]
        if start_char in puncs:
            data[i-1] += start_char
            data[i] = data[i][1:]
    
    return data

def batch_strip(data: list[str]) -> list[str]:
    new_data = []
    for item in data:
        if item == "":
            continue
        new_data.append(item.strip())
    
    return data

def chunk_sky_files(
        input_folder:str, 
        output_folder:str, 
        chunk_size:int=256,
        overlap_size:int=0,
    ) -> None:
    separators = ["。", "！", "？", "\n\n", "\n", "；", " "]
    print(f"Load tokenizer from tiktoken encoder...")
    r_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=separators,
        keep_separator=True
    )

    os.makedirs(output_folder, exist_ok=True)
    file_list = os.listdir(input_folder)
    file_list = [x for x in file_list if x.endswith(".parquet") and x.startswith("sky_")]
    print(f"file_nums: {len(file_list)}")

    for id_file, file in enumerate(file_list):
        print(f"> Process file {id_file}: {file}")

        origin_file = os.path.join(input_folder, file)
        output_file = os.path.join(
            output_folder, file.replace(".parquet", ".chunk_256.parquet")
        )

        parquet_table = pq.read_table(origin_file)
        print(f"data_len={parquet_table.num_rows}")
        chunked_data = []
        chunk_cnt = 0
        for text in progress.track(parquet_table['text'],total=parquet_table.num_rows):

            text = text.as_py()

            chunked_text = r_splitter.split_text(text)
            chunked_text = batch_move_ending_punc(chunked_text)
            chunked_text = batch_strip(chunked_text)

            chunked_data.extend(chunked_text)
            chunk_cnt += len(chunked_text)
        
        tb = pa.Table.from_arrays([pa.array(chunked_data)], names=["text"])
        pq.write_table(
            table=tb,
            where=output_file,
            row_group_size=50000,
            data_page_size=50000
        )
        print(f"Chunk data saved to: {output_file}")

        # head_1000_path = os.path.join(
        #     output_folder, file.replace(".parquet", ".chunk_256_head_1000.jsonl")
        # )
        # head_1000 = chunked_data[:1000]
        # head_1000 = [{"text":x} for x in head_1000]
        # write_jsonl_file(head_1000_path, head_1000)


def count_many_parquet_data(input_folder:str) -> None:
    file_list = os.listdir(input_folder)
    file_list = [x for x in file_list if x.endswith(".parquet") and x.startswith("sky_")]
    print(f"file_nums: {len(file_list)}")

    chunk_cnt = 0
    for id_file, file in enumerate(file_list):
        print(f"> Count data in file {id_file}: {file}")

        parquet_table = pq.read_table(os.path.join(input_folder, file))
        print(f"curr_data_len={parquet_table.num_rows}")
        chunk_cnt += parquet_table.num_rows
    
    print(f"\nTotal_data_len = {chunk_cnt}")

def sample_head(sample_num:int=1000) -> None:
    input_folder = "Chinese-corpus/my_chunk_256_data"
    filename = "sky_2020-40_zh_head_0000.chunk_256.parquet"

    load_path = os.path.join(input_folder, filename)

    tb = pq.read_table(load_path)
    sampled_data = tb['text'][:sample_num]
    sampled_data = [{"text": x.as_py()} for x in sampled_data]

    save_path = os.path.join(
        input_folder,
        filename.replace(".parquet", "_head_1000.jsonl")
    )
    write_jsonl_file(save_path, sampled_data)

def add_chunk_id_to_parquet_data(input_folder:str) -> None:
    file_list = os.listdir(input_folder)
    file_list = [x for x in file_list if x.endswith(".parquet") and x.startswith("sky_")]
    print(f"file_nums: {len(file_list)}")

    cnt_all_chunks = 0
    for id_file, file in enumerate(file_list):
        print(f"> Process file {id_file}: {file}")

        data_path = os.path.join(input_folder, file)
        parquet_table = pq.read_table(data_path)

        data_with_ids = []

        for text in progress.track(parquet_table["text"], total=parquet_table.num_rows):
            text = text.as_py()
            data_with_ids.append({"id":f"chunk_{cnt_all_chunks}", "text":text})
            cnt_all_chunks += 1
        
        df = pd.DataFrame(data_with_ids)
        new_parquet_table = pa.Table.from_pandas(df)

        # replace original data file with new data
        pq.write_table(new_parquet_table, data_path)
        print(f"New data saved to {data_path}")
        print(f"cnt_curr_chunks = {parquet_table.num_rows}")


        # check each data file
        head_100_path = os.path.join(
            input_folder, file.replace(".parquet", "_head_100.jsonl")
        )
        head_100 = data_with_ids[:100]
        write_jsonl_file(head_100_path, head_100)
    
    print(f"\nTotal_data_len = {cnt_all_chunks}")


def sample_from_sky_chunk_data(sample_prob:float, input_folder:str, output_path:str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_list = os.listdir(input_folder)
    file_list = [x for x in file_list if x.endswith(".parquet") and x.startswith("sky_")]
    print(f"file_nums: {len(file_list)}")

    # # count total chunk number
    # chunk_cnt = 0
    # for id_file, file in enumerate(file_list):
    #     print(f"> Count data in file {id_file}: {file}")

    #     parquet_table = pq.read_table(os.path.join(input_folder, file))
    #     print(f"curr_data_len={parquet_table.num_rows}")
    #     chunk_cnt += parquet_table.num_rows
    
    # print(f"\nTotal_data_len = {chunk_cnt}")

    # sample from all chunks
    sampled_chunks = []
    for id_file, file in enumerate(file_list):
        print(f"> Sample from file {id_file}: {file}")

        origin_file = os.path.join(input_folder, file)

        parquet_table = pq.read_table(origin_file)
        for id, text in progress.track(zip(parquet_table['id'],parquet_table['text']),total=parquet_table.num_rows):
            id, text = id.as_py(), text.as_py()

            rand_f = random.random()
            if rand_f <sample_prob:
                sampled_chunks.append({"id":id, "text":text})

        print(f"# curr_sampled_chunks = {len(sampled_chunks)}")
    
    # save parquet data
    df = pd.DataFrame(sampled_chunks)
    sampled_parquet_table = pa.Table.from_pandas(df)
    pq.write_table(table=sampled_parquet_table, where=output_path)
    print(f"Sampled {len(sampled_chunks)} chunks saved to {output_path}")

    # save jsonl data
    jsonl_path = output_path.replace(".parquet", ".jsonl")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in progress.track(sampled_chunks, description="write jsonl"):
            f.write(ujson.dumps(item, ensure_ascii=False)+"\n")
    print(f"Sampled {len(sampled_chunks)} chunks saved to {jsonl_path}")

if __name__ == "__main__":
    # parquet_to_jsonl()

    # # 1. generate sky corpus chunks
    # chunk_sky_files(
    #     input_folder="data/corpus_data/my_data/SkyPile-150B",
    #     output_folder="data/corpus_data/my_chunk_256_data/SkyPile-150B",
    #     chunk_size=256
    # )

    # # 2. sample chunks for NER data construction
    # add_chunk_id_to_parquet_data(
    #     input_folder="data/corpus_data/my_chunk_256_data/SkyPile-150B"
    # )

    # 3. sample from all chunks
    sample_from_sky_chunk_data(
        sample_prob=0.5,
        input_folder="data/corpus_data/my_chunk_256_data/SkyPile-150B",
        output_path="data/corpus_data/my_chunk_256_data_sampled/sky_2020-40_zh_head_0000_head_10.chunk_256_sampled.jsonl"
    )