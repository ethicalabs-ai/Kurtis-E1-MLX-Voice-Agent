from kurtis_mlx import config
from kurtis_mlx.utils.llm import get_llm_response, translate_text
from kurtis_mlx.utils.stt import transcribe
import time
import numpy as np
import openai
import re
from rich.console import Console

console = Console()


def handle_response_and_playback(
    text,
    text_queue,
    _client_ignored, # Ignored, we create a fresh one
    history,
    _llm_model_ignored, # Ignored, we use config.LLM_MODEL
    max_tokens,
    translate,
    language,
    translation_model,
    llm_language="english",
):
    # Re-instantiate client to pick up any config changes (e.g. API URL from UI)
    # This is necessary because handle_interaction might have been called with an old client
    # while waiting for audio.
    client = openai.OpenAI(base_url=config.OPENAI_API_URL, api_key=config.OPENAI_API_KEY)
    llm_model = config.LLM_MODEL

    console.print("[green]Generating response...")
    try:
        response = get_llm_response(text, client, history, llm_model, max_tokens)
    except openai.NotFoundError:
        console.print(f"[bold red]Error: Model '{llm_model}' not found on server.[/bold red]")
        console.print("[yellow]Tip: Check your --llm-model argument or server configuration. Try using 'default_model'.[/yellow]")
        return
    except Exception as e:
        console.print(f"[bold red]Error generating response: {e}[/bold red]")
        return

    console.print(f"[cyan]Assistant: {response}")
    if translate and language != "english":
        response = translate_text(
            response,
            client,
            llm_language,
            language,
            config,
            translation_model=translation_model,
            max_tokens=max_tokens,
        )
        console.print(
            f"[magenta]Translated back to {config.SUPPORTED_LANGUAGES[language]['name']}: {response}"
        )
    text_queue.put(response)


def get_validated_transcription(audio_np, stt_model_name, sample_rate, whisper_backend="mlx", ggml_model_path=None):
    """
    Transcribes audio and validates the quality using Whisper's metadata.
    Returns the text if it's high quality, otherwise returns None.
    """
    console.print("[green]Transcribing...")
    # Get the full transcription result
    use_whisper_cpp = (whisper_backend == "whisper_cpp")
    transcription_result = transcribe(
        audio_np, 
        stt_model_name, 
        sample_rate=sample_rate,
        use_whisper_cpp=use_whisper_cpp,
        ggml_model_path=ggml_model_path
    )
    
    if use_whisper_cpp:
        # whisper-cpp-python returns a dict with 'text' field, but maybe not 'segments' in the same format?
        # The user example showed: {'text': '...'}
        # If response_format='verbose_json' is used, it might have segments.
        # Our stt.py wrapper returns whatever the library returns.
        # If it's just {'text': ...}, we can't do the same validation.
        # Let's check if 'segments' exists.
        if not isinstance(transcription_result, dict):
             # Should be a dict
             text = str(transcription_result).strip()
             return text
             
    text = transcription_result.get("text", "").strip()
    console.print(f"[dim]Raw transcription: '{text}'[/dim]")
    
    # Regex to remove artifacts like [Silence], (Music), etc.
    # Matches square brackets or parentheses containing text
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    text = text.strip()
    console.print(f"[dim]Cleaned transcription: '{text}'[/dim]")
    
    if not text or len(text) < 2:
        return None
        
    # Also check if text is just repeated characters or very short nonsense
    # We relax the length check to allow short commands like "Stop", "No", "Wait" (lengths 2-4)
    if len(text) < 2: 
         return None
    
    if len(text) < 4 and len(set(text)) < 2: # e.g. "AAA"
         console.print(f"[yellow]Filtered repetitive short text: {text}")
         return None

    if use_whisper_cpp:
        if "segments" not in transcription_result:
            # Skip validation if no segments, but we did text validation above
            return text

    # Check the quality
    avg_confidence = -1.0
    no_speech_prob = 1.0

    if "segments" in transcription_result and transcription_result["segments"]:
        try:
            # Get the average confidence (avg_logprob)
            segment_probs = [
                seg.get("avg_logprob", -1.0) for seg in transcription_result["segments"]
            ]
            avg_confidence = sum(segment_probs) / len(segment_probs)

            # Get the "no speech" probability from the first segment
            no_speech_prob = transcription_result["segments"][0].get(
                "no_speech_prob", 0.0
            )
        except (IndexError, TypeError, ZeroDivisionError):
            console.print("[yellow]Could not parse transcription metadata.")
            # Keep default values to fail the check

    console.print(
        f"[green]Transcription confidence: {avg_confidence:.2f}, No-speech prob: {no_speech_prob:.2f}"
    )

    # Define quality thresholds
    CONFIDENCE_THRESHOLD = -0.8  # Closer to 0 is better. -0.8 is a decent filter.
    NO_SPEECH_THRESHOLD = 0.6  # Anything over 60% is likely noise.

    # Handle low-quality transcriptions
    if avg_confidence < CONFIDENCE_THRESHOLD or no_speech_prob > NO_SPEECH_THRESHOLD:
        console.print(
            f"[yellow]Low confidence ({avg_confidence:.2f}) or high no-speech prob ({no_speech_prob:.2f}). Skipping response."
        )
        return None  # Ignore this transcription

    return text


def is_echo(text, history):
    """
    Checks if the text is likely an echo of the last assistant message.
    """
    if not history:
        return False
    
    # Find last assistant message
    last_assistant_msg = None
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_assistant_msg = msg["content"]
            break
            
    if not last_assistant_msg:
        return False
        
    # Normalize
    text_norm = text.lower().strip()
    last_msg_norm = last_assistant_msg.lower().strip()
    
    # Check for substring match (if the echo is a part of the message)
    # We require a significant overlap to avoid false positives on common words
    if len(text_norm) > 10 and text_norm in last_msg_norm:
        return True
        
    # Check for Levenshtein distance or similar if needed, but substring is a good start for echo
    # Also check if the last message contains the text
    
    return False


def handle_interaction(
    text_queue,
    sound_queue,
    transcription_queue,
    stt_model_name,
    client,
    history,
    llm_model,
    max_tokens,
    translate,
    language,
    translation_model,
    is_busy_event,
    interrupt_event,
    whisper_backend="mlx",
    ggml_model_path=None,
):
    TARGET_LANGUAGES = [
        lang for lang in config.SUPPORTED_LANGUAGES if lang != "english"
    ]
    audio_np = transcription_queue.get()
    if audio_np is None:  # Shutdown signal
        return

    # Check if we are currently speaking
    # Check if we are currently speaking
    if is_busy_event.is_set():
        console.print("[yellow]Speech detected while speaking. Validating...")
        # We don't set is_busy_event here because it's already set.
        # But we need to validate if it's true speech before interrupting.

    console.print("[green]Transcribing...")
    text = (
        get_validated_transcription(
            audio_np, 
            stt_model_name, 
            sample_rate=16000,
            whisper_backend=whisper_backend,
            ggml_model_path=ggml_model_path
        ) or ""
    )
    
    if not text.strip():
        if is_busy_event.is_set():
             # We were busy, but validation failed (noise/echo).
             console.print("[yellow]Ignored noise/echo during playback.")
             return

        console.print(
            "[red]No text transcribed. Please ensure your microphone is working."
        )
        return 
        # If we are here, we are not playing back (unless we were interrupted? No).
        # Wait, is_busy_event is set by sound_worker when playing.
        # mic_worker no longer pauses.
        # So if is_busy_event is set, it means sound_worker is playing.
        # If we are here, we got audio.
        
        # If validation failed and we were busy, we just return and let playback continue.
        return

    # Valid speech detected
    # Check for echo
    if is_echo(text, history):
        console.print(f"[yellow]Ignored echo: {text}")
        # If we were busy, we stay busy (ignore echo)
        # If we weren't busy, we just ignore it.
        return

    if is_busy_event.is_set():
        console.print(f"[bold red]Interruption detected: {text}[/bold red]")
        interrupt_event.set()
        
        # Wait for playback to stop (is_busy_event cleared by sound_worker)
        # We use a timeout to avoid hanging if something goes wrong
        start_wait = time.time()
        while is_busy_event.is_set():
            if time.time() - start_wait > 2.0: # 2 seconds timeout
                console.print("[yellow]Warning: Playback stop timeout. Forcing continue.")
                break
            time.sleep(0.05)
            
            time.sleep(0.05)
            
        # Clear queues again to ensure no late arrivals from TTS
        while not text_queue.empty():
            try:
                text_queue.get_nowait()
            except:
                break
        while not sound_queue.empty():
            try:
                sound_queue.get_nowait()
            except:
                break
            
        interrupt_event.clear()
        console.print("[yellow]Playback interrupted.")
    
    console.print(f"[red]Text: {text}")
    if translate and language in TARGET_LANGUAGES:
        text = translate_text(
            text,
            client,
            language,
            "english",  # LLM Language
            config,
            translation_model=translation_model,
            max_tokens=max_tokens,
        )
        console.print(f"[magenta]Translated to English: {text}")
    console.print(f"[yellow]You: {text}")

    handle_response_and_playback(
        text,
        text_queue,
        client,
        history,
        llm_model,
        max_tokens,
        translate,
        language,
        translation_model,
    )


def handle_sip_interaction(
    text_queue,
    sound_queue,
    transcription_queue,
    stt_model_name,
    client,
    history,
    llm_model,
    max_tokens,
    translate,
    language,
    translation_model,
    interrupt_event,
    whisper_backend="mlx",
    ggml_model_path=None,
):
    """
    A variation of handle_interaction that gets audio from a queue
    (fed by the sip_worker) instead of recording directly.
    """
    # This will block until the sip_worker puts audio in the queue
    audio_np = transcription_queue.get()
    if audio_np is None:  # Shutdown signal
        return

    console.print("[green]Transcribing incoming call audio...")
    # SIP audio is 8kHz
    # SIP audio is 8kHz
    text = get_validated_transcription(
        audio_np, 
        stt_model_name, 
        sample_rate=8000,
        whisper_backend=whisper_backend,
        ggml_model_path=ggml_model_path
    ) or ""

    if not text:
        console.print("[yellow]Transcription empty, waiting for more audio.[/yellow]")
        return

    # Check for echo
    if is_echo(text, history):
        console.print(f"[yellow]Ignored echo: {text}")
        return

    # For SIP, we assume any valid speech is an interruption if we are speaking.
    # But we don't have is_busy_event for SIP easily accessible here?
    # Actually, SIP doesn't use is_busy_event in the same way.
    # But we passed interrupt_event.
    # We can just set it. If we are not playing, it does nothing (except maybe skip next playback).
    # Ideally we should know if we are playing.
    # But setting it blindly is safe-ish: it stops current playback.
    interrupt_event.set()
    
    # Clear queues
    while not text_queue.empty():
        try:
            text_queue.get_nowait()
        except:
            break
    while not sound_queue.empty():
        try:
            sound_queue.get_nowait()
        except:
            break

    console.print(f"[yellow]Caller: {text}")

    if translate and language != "english":
        text = translate_text(
            text,
            client,
            language,
            "english",
            config,
            translation_model=translation_model,
            max_tokens=max_tokens,
        )
        console.print(f"[magenta]Translated to English: {text}")

    handle_response_and_playback(
        text,
        text_queue,
        client,
        history,
        llm_model,
        max_tokens,
        translate,
        language,
        translation_model,
    )
