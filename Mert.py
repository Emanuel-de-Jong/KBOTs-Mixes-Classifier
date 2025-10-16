import numpy as np
import subprocess
import tempfile
import torch
import os
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from sklearn.utils import resample

class Mert():
    CHUNK_LENGTH_SECONDS = 15.0
    MODEL_NAME = "m-a-p/MERT-v1-330M"
    ERROR_LOG_NAME = "error.log"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True).to(self.device).eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_NAME, trust_remote_code=True, use_fast=False)
    
    def load_audio_ffmpeg(self, path):
        resample_rate = self.processor.sampling_rate
        
        with tempfile.NamedTemporaryFile(suffix='.f32', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = subprocess.run([
                "ffmpeg",
                "-y",
                "-i", path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ac", "1",
                "-ar", "24000",
                temp_path
            ], capture_output=True)

            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='ignore')
                raise Exception(f"FFmpeg error: {error_msg}")
            
            with open(temp_path, 'rb') as f:
                raw_data = f.read()
            
            audio_data = np.frombuffer(raw_data, dtype=np.float32).copy()
            
            waveform = torch.from_numpy(audio_data).float()
            return waveform, resample_rate
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def run(self, path, max_chunks=-1):
        print(f"Processing: {os.path.basename(path)}")
        
        try:
            audio_samples, resample_rate = self.load_audio_ffmpeg(path)
            
            samples_per_chunk = int(self.CHUNK_LENGTH_SECONDS * resample_rate)
            
            start_skip_samples = int(20.0 * resample_rate)
            end_skip_samples = int(20.0 * resample_rate)
            
            total_samples = len(audio_samples)
            usable_samples = total_samples - start_skip_samples - end_skip_samples
            
            if usable_samples < samples_per_chunk:
                self.error(f"{path} is too short after skipping! Usable: {usable_samples}, needed: {samples_per_chunk}")
                return None
            
            num_full_chunks = usable_samples // samples_per_chunk
            
            chunks = []
            for i in range(num_full_chunks):
                start_idx = start_skip_samples + (i * samples_per_chunk)
                end_idx = start_idx + samples_per_chunk
                chunk = audio_samples[start_idx:end_idx]
                chunks.append(chunk.numpy())

            if max_chunks != -1 and len(chunks) > max_chunks:
                chunks = resample(chunks, replace=False, n_samples=max_chunks, random_state=1)
            
            chunk_data = []
            for chunk in chunks:
                inputs = self.processor(chunk, sampling_rate=resample_rate, return_tensors="pt").to(self.device)

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
