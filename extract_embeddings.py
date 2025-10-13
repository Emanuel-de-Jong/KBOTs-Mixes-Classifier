import pandas as pd
import numpy as np
import torchaudio
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

CHUNK_LENGTH_SECONDS = 2.0

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "m-a-p/MERT-v1-330M"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to(device).eval()
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

df = pd.read_csv("labels.csv")
embs, labels = [], []

i = 0
for _, row in tqdm(df.iterrows(), total=len(df)):
    i += 1
    if i > 5:
        break
    
    path = row.filepath
    waveform, sr = torchaudio.load(path)

    resample_rate = processor.sampling_rate
    if sr != resample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate).cpu()
        waveform = resampler(waveform.cpu())

    audio_samples = waveform.squeeze(0)
    samples_per_chunk = int(CHUNK_LENGTH_SECONDS * resample_rate)
    
    total_samples = len(audio_samples)
    num_full_chunks = total_samples // samples_per_chunk
    
    print(f"Audio length: {total_samples/sr:.2f}s, Total samples: {total_samples}, Full chunks: {num_full_chunks}")
    
    chunk_embeddings = []
    for i in range(num_full_chunks):
        start_idx = i * samples_per_chunk
        end_idx = (i + 1) * samples_per_chunk
        chunk = audio_samples[start_idx:end_idx]
        
        input_audio_chunk = chunk.numpy()

        inputs = processor(input_audio_chunk, sampling_rate=resample_rate, return_tensors="pt")
        inputs = {k: v.half().to(device) if v.dtype == torch.float32 else v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        chunk_vec = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy().squeeze()
        chunk_embeddings.append(chunk_vec)
        
        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache()
    
    del waveform, audio_samples
    
    if chunk_embeddings:
        file_vec = np.mean(chunk_embeddings, axis=0) 
        embs.append(file_vec)
        labels.append(row.label)
        print(f"Successfully processed {len(chunk_embeddings)} chunks")

X = np.stack(embs)
pd.Series(labels).to_csv("y_labels.csv", index=False)
np.save("X_emb.npy", X)
print("Saved X_emb.npy and y_labels.csv")
