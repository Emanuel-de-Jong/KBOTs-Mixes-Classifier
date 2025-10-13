import pandas as pd
import numpy as np
import torchaudio
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

CHUNK_LENGTH_SECONDS = 2.0

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "m-a-p/MERT-v1-330M"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

df = pd.read_csv("labels.csv")
embs, labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row.filepath
    waveform, sr = torchaudio.load(path)

    resample_rate = processor.sampling_rate
    if sr != resample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate).cpu()
        waveform = resampler(waveform.cpu())

    audio_samples = waveform.squeeze(0)
    samples_per_chunk = int(CHUNK_LENGTH_SECONDS * resample_rate)
    
    chunks = torch.split(audio_samples, samples_per_chunk)
    chunk_embeddings = []
    for chunk in chunks:
        input_audio_chunk = chunk.numpy()
        inputs = processor(
            input_audio_chunk, 
            sampling_rate=resample_rate, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        chunk_vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        chunk_embeddings.append(chunk_vec)
        
        del inputs, outputs
        if device == "cuda":
            torch.cuda.empty_cache() 
    
    del waveform, audio_samples, chunks
    
    file_vec = np.mean(chunk_embeddings, axis=0) 
    
    embs.append(file_vec)
    labels.append(row.label)

X = np.stack(embs)
pd.Series(labels).to_csv("y_labels.csv", index=False)
np.save("X_emb.npy", X)
print("Saved X_emb.npy and y_labels.csv")
