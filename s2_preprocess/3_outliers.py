import pandas as pd
import numpy as np
import joblib
import gc
import os
import s0_utils.global_params as g

STEP = 3
MAX_REMOVE_PERC = 0.1
MIN_SONG_ROWS = 10

g.DATA_BATCH_SIZE = 3_000

# Per label/playlist/genre (all synonyms) remove rows where the features are too far outside the normal features of the other rows in the same label.
# Make sure to never remove more than MAX_REMOVE_PERC (where 1.0 would allow removing everything) of all rows in a label.
# Make sure to never remove so many rows that there are less than MIN_SONG_ROWS rows of a song.
for data_set_type in g.DataSetType:
    out_idx = 0
    out_rows = []
    for data_path in g.iter_data_paths(STEP-1, data_set_type):
        data = g.load_data(data_path)

        for _, row in data.iterrows():
            out_rows.append(row)
            if len(out_rows) >= g.DATA_BATCH_SIZE:
                g.save_data(pd.DataFrame(out_rows), STEP, data_set_type, out_idx)
                out_idx += 1
                out_rows = []
