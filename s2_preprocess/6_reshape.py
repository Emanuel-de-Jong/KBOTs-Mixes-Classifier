import gc
import shutil
import numpy as np
import pandas as pd
import zarr
import s0_utils.global_params as config
from tqdm import tqdm

SHOULD_PASS = False

def reshape_data(feature_data):
    feature_data = feature_data[
        :,
        :config.DATA_COUNTS[config.DataSectionType.time],
        :config.DATA_COUNTS[config.DataSectionType.feature],
        config.DATA_LAYER_START:config.DATA_LAYER_END + 1
    ]
    
    axis_mapping = {
        config.DataSectionType.time: 1,
        config.DataSectionType.feature: 2,
        config.DataSectionType.layer: 3
    }
    
    transpose_order = [0]
    for section in config.DATA_ORDER:
        transpose_order.append(axis_mapping[section])
    
    return np.transpose(feature_data, transpose_order)

for data_set_type in tqdm(config.DataSetType, desc="Processing data sets"):
    current_batch_index = 0

    for source_zarr_path in tqdm(
        config.iter_zarr_data_paths(5, data_set_type),
        desc="Processing source batches",
        leave=False
    ):
        if SHOULD_PASS:
            destination_path = config.CACHE_DIR / source_zarr_path.name.replace("data_5_", "data_6_")
            shutil.copytree(source_zarr_path, destination_path)
            continue

        source_zarr = zarr.open(source_zarr_path, mode="r")
        total_samples = source_zarr["label"].shape[0]

        for chunk_start in range(0, total_samples, config.DATA_BATCH_SIZE):
            chunk_end = min(chunk_start + config.DATA_BATCH_SIZE, total_samples)

            feature_data = source_zarr["data"][chunk_start:chunk_end]
            label_data = source_zarr["label"][chunk_start:chunk_end]

            reshaped_features = reshape_data(feature_data)

            batch_dataframe = pd.DataFrame({
                "data": list(reshaped_features),
                "label": label_data
            })

            config.save_zarr(batch_dataframe, 6, data_set_type, current_batch_index)
            current_batch_index += 1

            del feature_data, label_data, reshaped_features, batch_dataframe
            gc.collect()

        del source_zarr
        gc.collect()
