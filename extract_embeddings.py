import pandas as pd
import numpy as np
import torchaudio
import torch
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "m-a-p/MERT-v1-330M"
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

df = pd.read_csv("labels.csv")
embs, labels = [], []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = row.filepath
    waveform, sr = torchaudio.load(path)
    
    resample_rate = processor.sampling_rate
    if sr != resample_rate:
        resampler = torchaudio.transforms.Resample(sr, resample_rate)
        waveform = resampler(waveform)
    
    input_audio = waveform.squeeze(0).numpy()
    inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
    embs.append(vec)
    labels.append(row.label)

X = np.stack(embs)
pd.Series(labels).to_csv("y_labels.csv", index=False)
np.save("X_emb.npy", X)
print("Saved X_emb.npy and y_labels.csv")
