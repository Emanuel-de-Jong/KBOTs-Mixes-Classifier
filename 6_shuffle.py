import gc
import pandas as pd
import global_params as g

g.DATA_BATCH_SIZE = 14_000

for data_set_type in g.DataSetType:
    dfs = []
    for data_path in g.iter_data_paths(5, data_set_type):
        dfs.append(g.load_data(data_path))

    data = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    data = data.sample(frac=1).reset_index(drop=True)
    g.save_data_batched(data, 6, data_set_type)
    del data
    gc.collect()
