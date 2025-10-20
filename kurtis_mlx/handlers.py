from kurtis_mlx import config
from kurtis_mlx.utils.sound import record_until_enter
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


def handle_interaction(
    text_queue,
    stt_model_name,
    client,
    history,
    llm_model,
    max_tokens,
    samplerate,
    translate,
    language,
    translation_model,
):
    TARGET_LANGUAGES = [
        lang for lang in config.SUPPORTED_LANGUAGES if lang != "english"
    ]
    audio_np = record_until_enter(samplerate)
    if audio_np.size == 0:
        console.print(
            "[red]No audio recorded. Please ensure your microphone is working."
        )
        return

    console.print("[green]Transcribing...")
    text = transcribe(audio_np, stt_model_name, sample_rate=samplerate)
    if not text.strip():
        console.print(
            "[red]No text transcribed. Please ensure your microphone is working."
        )
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
    text = transcribe(audio_np, stt_model_name, sample_rate=8000)

    if not text.strip():
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
