import numpy as np
import torchaudio
import torch
import os
from transformers import AutoModel, Wav2Vec2FeatureExtractor

class Mert():
    CHUNK_LENGTH_SECONDS = 15.0
    MODEL_NAME = "m-a-p/MERT-v1-330M"
    ERROR_LOG_NAME = "error.log"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.device).eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_fast=False)
    
    def run(self, path):
        print(f"Processing: {os.path.basename(path)}")
        
        try:
            waveform, sr = torchaudio.load(path)
            # if waveform[0][5] == 0 and waveform[0][25] == 0 and waveform[0][50] == 0 and \
            #         waveform[1][5] == 0 and waveform[1][25] == 0 and waveform[1][50] == 0:
            #     self.error(f"{path} is corrupt! Only 0s.")
            #     return None
            
            resample_rate = self.processor.sampling_rate
            if sr != resample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=resample_rate).cpu()
                waveform = resampler(waveform.cpu())
            
            if waveform.dim() > 1:
                audio_samples = waveform.mean(dim=0)
            else:
                audio_samples = waveform.squeeze(0)
            
            samples_per_chunk = int(self.CHUNK_LENGTH_SECONDS * resample_rate)
            
            start_skip_samples = int(20.0 * resample_rate)
            end_skip_samples = int(20.0 * resample_rate)
            
            total_samples = len(audio_samples)
            
            usable_samples = total_samples - start_skip_samples - end_skip_samples
            
            if usable_samples < samples_per_chunk:
                self.error(f"{path} is too short after skipping! Usable: {usable_samples}, needed: {samples_per_chunk}")
                return None
            
            num_full_chunks = usable_samples // samples_per_chunk
            
            chunk_data = []
            for i in range(num_full_chunks):
                start_idx = start_skip_samples + (i * samples_per_chunk)
                end_idx = start_idx + samples_per_chunk
                chunk = audio_samples[start_idx:end_idx]
                
                input_audio_chunk = chunk.numpy()

                inputs = self.processor(input_audio_chunk, sampling_rate=resample_rate, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                chunk_vec = outputs.last_hidden_state.mean(dim=1).cpu().float().numpy().squeeze()
                chunk_data.append(chunk_vec)
            
            print(f"Success! Generated {len(chunk_data)} chunks")
            return chunk_data
        
        except Exception as e:
            self.error(f"{path} is corrupt! Error: {e}")
            return None

    def error(self, message):
        print(message)

        with open(self.ERROR_LOG_NAME, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
