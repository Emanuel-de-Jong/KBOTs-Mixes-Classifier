import pandas as pd
import numpy as np
import torchaudio
import torch
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "m-a-p/MERT-v1-330M"
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

df = pd.read_csv("labels.csv")
embs, labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row.filepath
    waveform, sr = torchaudio.load(path)
    if sr != 24000:  # MERT expects 24 kHz
        waveform = torchaudio.functional.resample(waveform, sr, 24000)
    inputs = processor(audios=waveform.squeeze(0), sampling_rate=24000, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    
    # Take mean of last hidden layer embeddings
    vec = out.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
    embs.append(vec)
    labels.append(row.label)

X = np.stack(embs)
pd.Series(labels).to_csv("y_labels.csv", index=False)
np.save("X_emb.npy", X)
print("Saved X_emb.npy and y_labels.csv")
