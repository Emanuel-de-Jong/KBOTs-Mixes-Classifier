import numpy as np
import subprocess
import tempfile
import torch
import os
import random
import s0_utils.global_params as g
from transformers import AutoModel, Wav2Vec2FeatureExtractor

class Mert():
    START_SKIP_SECONDS = 0
    END_SKIP_SECONDS = 0

    CHUNK_LENGTH_SECONDS = 1
    WINDOW_LENGTH_SECONDS = CHUNK_LENGTH_SECONDS * (30 // CHUNK_LENGTH_SECONDS)
    MIN_WINDOW_LENGTH_SECONDS = 5

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
    
    def volume_normalize(self, samples, target_dbfs=-20.0, eps=1e-8):
        rms = torch.sqrt(torch.mean(samples ** 2) + eps)
        target_rms = 10 ** (target_dbfs / 20.0)
        gain = target_rms / rms
        return samples * gain
    
    def run(self, path, max_chunks=-1, max_windows=-1):
        # print(f"Processing: {os.path.basename(path)}")
        
        try:
            samples, resample_rate = self.load_audio_ffmpeg(path)
            samples = self.volume_normalize(samples)

            start_skip_samples = int(self.START_SKIP_SECONDS * resample_rate)
            end_skip_samples = int(self.END_SKIP_SECONDS * resample_rate)
            samples = samples[start_skip_samples:len(samples)-end_skip_samples]
            sample_count = len(samples)
            
            samples_min_window = int(self.MIN_WINDOW_LENGTH_SECONDS * resample_rate)
            if sample_count < samples_min_window:
                self.error(f"{path} is too short after skipping! Usable: {sample_count}, needed: {samples_min_window}")
                return None
            
            samples_per_window = int(self.WINDOW_LENGTH_SECONDS * resample_rate)
            samples_per_chunk = int(self.CHUNK_LENGTH_SECONDS * resample_rate)

            windows = []
            window_count = (sample_count // samples_per_window) + 1
            for i in range(window_count):
                start_idx = i * samples_per_window
                end_idx = min(start_idx + samples_per_window, sample_count)
                if end_idx - start_idx < samples_min_window:
                    break

                end_idx = (end_idx // samples_per_chunk) * samples_per_chunk

                windows.append(samples[start_idx:end_idx].numpy())

            if max_windows != -1:
                random.shuffle(windows)
                windows = windows[:max_windows]

            embs = []
            for window in windows:
                inputs = self.processor(window, sampling_rate=resample_rate, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                
                out = torch.stack(outputs.hidden_states).squeeze().cpu()
                out = out[g.DATA_LAYER_INDEXES]

                out = torch.nn.functional.adaptive_avg_pool1d(
                    out.permute(0, 2, 1), output_size=len(window) // samples_per_chunk
                ).permute(2, 0, 1) # Always ordered (time, layer, feature)

                for emb in out:
                    embs.append(emb.unsqueeze(0).numpy())

            random.shuffle(embs)
            if max_chunks != -1:
                embs = embs[:max_chunks]
            
            # print(f"Success! Generated {len(embs)} embs")
            return np.array(embs)
        
        except Exception as e:
            self.error(f"{path} is corrupt! Error: {e}")
            return None

    def error(self, message):
        print(message)

        with open(self.ERROR_LOG_NAME, "a", encoding="utf-8") as f:
            f.write(f"{message}\n")
