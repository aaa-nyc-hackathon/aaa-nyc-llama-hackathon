import ffmpeg
import numpy as np
import torch

def extract_audio_tensor_from_mp4(path, sample_rate=16000):
    # Extract raw mono PCM 16-bit audio
    out, _ = (
        ffmpeg
        .input(path)
        .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar=sample_rate)
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Convert to NumPy int16 array
    audio_np = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0  # Normalize to [-1.0, 1.0]

    # Convert to PyTorch tensor
    audio_tensor = torch.from_numpy(audio_np)

    return audio_tensor  # shape: [samples]

# Example usage
audio_tensor = extract_audio_tensor_from_mp4("alabama_clemson_30s_clip.mp4")
print(audio_tensor.shape, audio_tensor.max(), audio_tensor.argmax(), audio_tensor.argmax().item() / audio_tensor.shape[0])
