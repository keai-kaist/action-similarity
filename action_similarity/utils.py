import os
import pickle
import re

def parse_action_label(action_label):
    actions = {}
    with open(action_label) as f:
        lines = f.readlines()
        for line in lines:
            no, action = line.split(None, maxsplit=1)
            no = int(re.search(r'\d+', no).group())
            actions[no] = action.strip()
    return actions

def cache_file(file_name: str, func, *args, **kargs):
    file_name = file_name.rstrip(".mp4") + ".pickle"
    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
    else:
        data = func(*args, **kargs)
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
    return data