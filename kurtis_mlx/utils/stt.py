import librosa
import numpy as np
import sys
import mlx_whisper

try:
    from whisper_cpp_python import Whisper
    HAS_WHISPER_CPP = True
except ImportError:
    HAS_WHISPER_CPP = False


TARGET_SAMPLE_RATE = 16000


_whisper_cpp_instance = None

def transcribe(audio_np, stt_model_name, sample_rate=TARGET_SAMPLE_RATE, use_whisper_cpp=False, ggml_model_path=None):
    """
    Transcribes audio to text using mlx-whisper or whisper-cpp-python.
    The sample rate of the audio must be provided.
    """
    global _whisper_cpp_instance
    # Whisper expects audio at 16kHz. We need to resample if it's different.
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_resampled = librosa.resample(
            np.asarray(audio_np, dtype=np.float32),
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
            res_type="soxr_vhq",  # Use a high-quality resampler
        ).astype(np.float32)
    else:
        audio_resampled = audio_np

    # This will now correctly normalize:
    # 1. The new 16kHz resampled float array (from 8kHz)
    # 2. Or the original 16kHz int16 array (in non-SIP mode)
    
    if use_whisper_cpp:
        if not HAS_WHISPER_CPP:
            raise ImportError("whisper-cpp-python is not installed.")
        if not ggml_model_path:
            raise ValueError("ggml_model_path must be provided when using whisper-cpp-python.")
            
        if _whisper_cpp_instance is None:
             _whisper_cpp_instance = Whisper(model_path=ggml_model_path)
        
        # whisper-cpp-python expects float32 audio
        audio_float32 = audio_resampled.astype(np.float32) / 32768.0
        
        # It seems whisper-cpp-python transcribe expects a file path or similar?
        # The user example showed: output = whisper.transcribe(open('samples/jfk.mp3'))
        # But looking at library source or common usage, usually it accepts numpy array too?
        # Let's assume it accepts numpy array or we might need to save to temp file.
        # WAIT, the user example: whisper.transcribe(open('samples/jfk.mp3'))
        # If it strictly requires file-like object, we can wrap numpy in BytesIO?
        # Actually, standard whisper.cpp bindings usually accept numpy.
        # Let's try passing the numpy array directly first.
        # If it fails, we might need to check the library docs or source.
        # Re-reading user request: "output = whisper.transcribe(open('samples/jfk.mp3'))"
        # It takes a file object.
        # Let's try to save to a temporary wav file to be safe, or use a BytesIO if supported.
        # For robustness, let's save to a temp file.
        
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, audio_float32, TARGET_SAMPLE_RATE)
            # Re-open as file object as per example
            # output = _whisper_cpp_instance.transcribe(open(tmp.name, 'rb'))
            # Actually, usually passing the path string works too.
            # Let's try passing the path first, if not open it.
            # The user example passed an open file object.
            output = _whisper_cpp_instance.transcribe(open(tmp.name, "rb"))
            
        return output
        
    else:
        return mlx_whisper.transcribe(
            audio_resampled.astype(np.float32) / 32768.0,
            fp16=False,
            path_or_hf_repo=stt_model_name,
        )
