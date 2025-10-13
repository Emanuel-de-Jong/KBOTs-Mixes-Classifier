import numpy as np
import torchaudio
import torch
import os
from transformers import AutoModel, Wav2Vec2FeatureExtractor

class Mert():
    CHUNK_LENGTH_SECONDS = 10.0
    MODEL_NAME = "m-a-p/MERT-v1-330M"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16) \
            .to(self.device) \
            .eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_fast=False)
    
    def run(self, path):
        print(f"Processing: {os.path.basename(path)}")
        
        waveform, sr = torchaudio.load(path)

        resample_rate = self.processor.sampling_rate
        if sr != resample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate).cpu()
            waveform = resampler(waveform.cpu())
        
        if waveform.dim() > 1:
            audio_samples = waveform.mean(dim=0)
        else:
            audio_samples = waveform.squeeze(0)
        
        samples_per_chunk = int(self.CHUNK_LENGTH_SECONDS * resample_rate)
        
        total_samples = len(audio_samples)
        num_full_chunks = total_samples // samples_per_chunk
        
        chunk_vecs = []
        for i in range(num_full_chunks):
            start_idx = i * samples_per_chunk
            end_idx = (i + 1) * samples_per_chunk
            chunk = audio_samples[start_idx:end_idx]
            
            input_audio_chunk = chunk.numpy()

            inputs = self.processor(input_audio_chunk, sampling_rate=resample_rate, return_tensors="pt")
            inputs = {k: v.half().to(self.device) if v.dtype == torch.float32 else v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            chunk_vec = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy().squeeze()
            chunk_vecs.append(chunk_vec)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return np.mean(chunk_vecs, axis=0)