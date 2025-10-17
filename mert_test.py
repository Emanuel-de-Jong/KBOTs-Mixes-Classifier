from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import numpy as np
import torch
import subprocess
import tempfile
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).to(device)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)

def load_audio_ffmpeg(path):
    resample_rate = processor.sampling_rate
    
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

input_audio, resample_rate = load_audio_ffmpeg("train/Acid Techno/Deniz Kabu - In A Life.mp3")
  
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

# take a look at the output shape, there are 25 layers of representation
# each layer performs differently in different downstream tasks, you should choose empirically
all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [25 layer, Time steps, 1024 feature_dim]

# for utterance level classification tasks, you can simply reduce the representation in time
time_reduced_hidden_states = all_layer_hidden_states.mean(-2)
print(time_reduced_hidden_states.shape) # [25, 1024]
