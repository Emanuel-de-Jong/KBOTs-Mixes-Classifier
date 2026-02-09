import gc
import shutil
import numpy as np
import zarr
import s0_utils.global_params as g

SHOULD_PASS = False

def reshape_data(feature_data):
    feature_data = feature_data[
        :,
        :g.DATA_COUNTS[g.DataSectionType.time],
        :g.DATA_COUNTS[g.DataSectionType.feature],
        :g.DATA_COUNTS[g.DataSectionType.layer]
    ]
    
    axis_mapping = {
        g.DataSectionType.time: 1,
        g.DataSectionType.feature: 2,
        g.DataSectionType.layer: 3
    }
    
    transpose_order = [0]
    for section in g.DATA_ORDER:
        transpose_order.append(axis_mapping[section])
    
    return np.transpose(feature_data, transpose_order)

for data_set_type in g.DataSetType:
    current_batch_index = 0

    for source_zarr_path in g.iter_zarr_data_paths(5, data_set_type):
        if SHOULD_PASS:
            destination_path = g.CACHE_DIR / source_zarr_path.name.replace("data_5_", "data_6_")
            shutil.copytree(source_zarr_path, destination_path)
            current_batch_index += 1
            continue

        source_zarr = zarr.open(source_zarr_path, mode="r")
        total_samples = source_zarr["label"].shape[0]

        destination_zarr = zarr.open_group(
            g.CACHE_DIR / f"data_6_{data_set_type.name}_{current_batch_index}.zarr",
            mode="w"
        )

        first = True

        for chunk_start in range(0, total_samples, g.DATA_BATCH_SIZE):
            chunk_end = min(chunk_start + g.DATA_BATCH_SIZE, total_samples)

            feature_data = source_zarr["data"][chunk_start:chunk_end]
            label_data = source_zarr["label"][chunk_start:chunk_end]

            reshaped_features = reshape_data(feature_data)

            if first:
                destination_zarr.create_array(
                    name="data",
                    data=reshaped_features,
                    chunks=(min(g.MODEL_BATCH_SIZE, len(reshaped_features)),) + reshaped_features.shape[1:],
                    compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")],
                )

                destination_zarr.create_array(
                    name="label",
                    data=label_data,
                    chunks=(min(g.MODEL_BATCH_SIZE, len(label_data)),),
                    compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")],
                )

                first = False
            else:
                destination_zarr["data"].append(reshaped_features)
                destination_zarr["label"].append(label_data)

            del feature_data, label_data, reshaped_features
            gc.collect()

        del source_zarr, destination_zarr
        gc.collect()

        current_batch_index += 1
