from kurtis_mlx import config
from kurtis_mlx.utils.llm import get_llm_response, translate_text
from kurtis_mlx.utils.stt import transcribe
from rich.console import Console

console = Console()


def handle_response_and_playback(
    text,
    text_queue,
    client,
    history,
    llm_model,
    max_tokens,
    translate,
    language,
    translation_model,
    llm_language="english",
):
    console.print("[green]Generating response...")
    response = get_llm_response(text, client, history, llm_model, max_tokens)
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


def get_validated_transcription(audio_np, stt_model_name, sample_rate):
    """
    Transcribes audio and validates the quality using Whisper's metadata.
    Returns the text if it's high quality, otherwise returns None.
    """
    console.print("[green]Transcribing...")
    # 1. Get the full transcription result
    transcription_result = transcribe(audio_np, stt_model_name, sample_rate=sample_rate)
    text = transcription_result.get("text", "").strip()

    # 2. Check the quality
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

    if not text:
        console.print("[red]No text transcribed.")
        return None

    console.print(f"[red]Text: {text}")
    return text


def handle_interaction(
    text_queue,
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
):
    TARGET_LANGUAGES = [
        lang for lang in config.SUPPORTED_LANGUAGES if lang != "english"
    ]
    audio_np = transcription_queue.get()
    if audio_np is None:  # Shutdown signal
        return

    is_busy_event.set()

    console.print("[green]Transcribing...")
    text = get_validated_transcription(audio_np, stt_model_name, sample_rate=16000)
    if not text:
        console.print(
            "[red]No text transcribed. Please ensure your microphone is working."
        )
        is_busy_event.clear()
        return
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
    transcription_queue,
    stt_model_name,
    client,
    history,
    llm_model,
    max_tokens,
    translate,
    language,
    translation_model,
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
    text = get_validated_transcription(audio_np, stt_model_name, sample_rate=8000)

    if not text:
        console.print("[yellow]Transcription empty, waiting for more audio.[/yellow]")
        return

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
