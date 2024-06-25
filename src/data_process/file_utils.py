import json

def load_data(path, json_format=None):
    if path.endswith(".txt"):
        data = [eval(x.strip()) for x in open(path, "r", encoding="utf-8").readlines()]    
    elif path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
    elif path.endswith(".jsonl"):
        with open(path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
    else:
        raise ValueError(f"Wrong path for query data: {path}")

    return data

def save_data(path, data, indent=4):
    if path.endswith(".txt"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data[:-1]:
                f.write(str(item) + "\n")
            f.write(str(data[-1]))
    elif path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, indent=indent, ensure_ascii=False))
    elif path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                line = json.dumps(item, ensure_ascii=False)
                f.write(line + "\n")
    else:
        raise ValueError(f"Wrong path for prompts saving: {path}")

def json2dict(json_list):
    d = dict()
    for item in json_list:
        k = list(item.keys())[0]
        v = item[k]
        d[k] = v
    return d

def dict2json(in_dict):
    out_json = []
    for k, v in in_dict.items():
        out_json.append(
            {k: v}
        )
    return out_json

def convert_format(data, target_format):
    if target_format == "dict":
        if isinstance(data, dict):
            data_new = data
        elif isinstance(data, list):
            data_new = json2dict(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    elif target_format == "json":
        if isinstance(data, list):
            data_new = data
        elif isinstance(data, dict):
            data_new = dict2json(data)
        else:
            raise TypeError(f"Cannot handle type(data)={type(data)}")
    else:
        raise NotImplementedError(f"convert_format() Not implemented for target_format={target_format}")

    return data_new