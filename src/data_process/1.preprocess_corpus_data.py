from os.path import dirname, abspath, exists, isdir
from os import remove, mkdir, walk
import os
import time
from unicodedata import normalize
import re
import pandas as pd

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ujson
from rich import progress
import fastparquet


def gen_sky(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in progress.track(os.listdir(input_folder)):
        if not filename.endswith(".jsonl"):
            continue
        origin_file = os.path.join(input_folder, filename)
        output_file = os.path.join(
            output_folder, "sky_"+filename.replace(".jsonl", ".parquet")
        )
        print(f"Processing {origin_file}...")

        lines = []
        with open(origin_file, "r", encoding="utf-8") as f:
            for line in f:
                item = ujson.loads(line)
                lines.append(item["text"])  # make sure each line is a JSON object

        if lines:  # make sure file is not empty
            tb = pa.Table.from_arrays([pa.array(lines)], names=["text"])
            pq.write_table(
                table=tb,
                where=output_file,
                row_group_size=50000,
                data_page_size=50000,
            )
            print(f"Processed {origin_file} to {output_file}")
        else:
            print(f"No content in {origin_file}. Skipping.")


if __name__ == "__main__":

    # gen_sky: only need to privide input dir and output dir
    gen_sky(
        input_folder="data/corpus_data/raw_data/SkyPile-150B", 
        output_folder="data/corpus_data/my_data/SkyPile-150B"
        )
