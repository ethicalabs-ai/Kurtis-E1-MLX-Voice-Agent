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
    TARGET_LANGUAGES = (
        [config.SUPPORTED_LANGUAGES[language]["name"]] if language != "english" else []
    )
    audio_np = record_until_enter(samplerate)
    if audio_np.size == 0:
        console.print(
            "[red]No audio recorded. Please ensure your microphone is working."
        )
        return

    console.print("[green]Transcribing...")
    text = transcribe(audio_np, stt_model_name)
    console.print(f"[red]Text: {text}")
    if translate and language in TARGET_LANGUAGES:
        text = translate_text(
            text,
            client,
            "english",  # LLM Language
            language,
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
