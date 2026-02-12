import gc
import shutil
import numpy as np
import zarr
import s0_utils.global_params as g
from pathlib import Path

def del_last_model():
    for path in g.CACHE_DIR.glob("data_6*"):
        shutil.rmtree(path, ignore_errors=True)

    shutil.rmtree(Path("s3_train") / "training", ignore_errors=True)

    test_log = Path("s3_train") / "2_test.log"
    if test_log.exists():
        test_log.unlink()

    model_path = g.MODELS_DIR / f"model_{g.NAME}.keras"
    if model_path.exists():
        model_path.unlink()

def reshape_data(feature_data):
    feature_data = feature_data[
        :,
        :g.DATA_COUNTS[g.DataSectionType.time],
        :g.DATA_COUNTS[g.DataSectionType.layer],
        :g.DATA_COUNTS[g.DataSectionType.feature]
    ]
    
    axis_mapping = {
        g.DataSectionType.time: 1,
        g.DataSectionType.layer: 2,
        g.DataSectionType.feature: 3
    }
    
    transpose_order = [0]
    for section in g.DATA_ORDER:
        transpose_order.append(axis_mapping[section])
    
    return np.transpose(feature_data, transpose_order)

print("Reshaping...")
# del_last_model()

for data_set_type in g.DataSetType:
    current_batch_index = 0

    for source_zarr_path in g.iter_zarr_data_paths(5, data_set_type):
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
