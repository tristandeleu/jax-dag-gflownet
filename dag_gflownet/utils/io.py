import numpy as np
import haiku as hk

from typing import Mapping


def save(filename, **trees):
    data = {}
    for prefix, tree in trees.items():
        if isinstance(tree, Mapping):
            tree = hk.data_structures.to_haiku_dict(tree)
            for module_name, name, value in hk.data_structures.traverse(tree):
                data[f'{prefix}/{module_name}/{name}'] = value
        else:
            data[prefix] = tree

    np.savez(filename, **data)


def load(filename, **kwargs):
    data = {}
    f = open(filename, 'rb') if isinstance(filename, str) else filename
    results = np.load(f, **kwargs)

    for key in results.files:
        prefix, delimiter, name = key.rpartition('/')
        if delimiter:
            prefix, _, module_name = prefix.partition('/')
            if prefix not in data:
                data[prefix] = {}
            if module_name not in data[prefix]:
                data[prefix][module_name] = {}
            data[prefix][module_name][name] = results[key]
        else:
            data[name] = results[key]

    for prefix, tree in data.items():
        if isinstance(tree, dict):
            data[prefix] = hk.data_structures.to_haiku_dict(tree)
    
    if isinstance(filename, str):
        f.close()
    del results

    return data