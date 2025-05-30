import click
from rich.console import Console
from openai import OpenAI
from multiprocessing import Process, Queue as MPQueue

from kurtis_mlx import config
from kurtis_mlx.workers.tts import tts_worker
from kurtis_mlx.workers.sound import sd_worker
from kurtis_mlx.handlers import handle_interaction


console = Console()


@click.command()
@click.option(
    "--language",
    default="english",
    type=click.Choice(config.SUPPORTED_LANGUAGES.keys()),
    help="Language for transcription and TTS.",
)
@click.option(
    "--speaker",
    type=click.Choice(config.SPEAKERS),
    help="Override default language speaker.",
)
@click.option(
    "--whisper-model",
    default="mlx-community/whisper-medium",
    help="Base Whisper model (combined with language code).",
)
@click.option(
    "--tts-model",
    default="multilingual/multi-dataset/xtts_v2",
    help="TTS model subpath",
)
@click.option("--max-tokens", default=200, help="Maximum tokens in LLM response.")
@click.option(
    "--samplerate", default=22050, help="Audio recording and playback sample rate."
)
@click.option(
    "--llm-model",
    default="hf.co/ethicalabs/Kurtis-E1.1-Qwen2.5-3B-Instruct-IQ4_XS-GGUF",
    help="LLM model identifier.",
)
@click.option(
    "--translate", is_flag=True, help="Translate assistant replies into user language."
)
@click.option(
    "--translation-model",
    default="hf.co/mradermacher/TowerInstruct-7B-v0.2-GGUF:Q4_K_S",
    help="Model to use for translation.",
)
def main(
    language,
    speaker,
    whisper_model,
    tts_model,
    max_tokens,
    samplerate,
    llm_model,
    translate,
    translation_model,
):
    history = [
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT,
        }
    ]

    lang_code = config.SUPPORTED_LANGUAGES[language]["code"]
    selected_speaker = (
        speaker or config.SUPPORTED_LANGUAGES[language]["default_speaker"]
    )
    full_whisper_model = whisper_model

    full_tts_model = tts_model

    client = OpenAI(base_url=config.OPENAI_API_URL, api_key=config.OPENAI_API_KEY)

    text_queue = MPQueue()
    sound_queue = MPQueue()

    tts_process = Process(
        target=tts_worker,
        args=(
            text_queue,
            sound_queue,
            full_tts_model,
            samplerate,
            lang_code,
            selected_speaker,
        ),
        daemon=True,
    )
    tts_process.start()
    sound_process = Process(
        target=sd_worker,
        args=(sound_queue, samplerate),
        daemon=True,
    )
    sound_process.start()

    try:
        while True:
            console.input("Press Enter to begin speaking...")
            handle_interaction(
                text_queue,
                full_whisper_model,
                client,
                history,
                llm_model,
                max_tokens,
                samplerate,
                translate,
                language,
                translation_model,
            )

    except KeyboardInterrupt:
        console.print("\n[red]KeyboardInterrupt. Exiting...")
        text_queue.put(None)
        sound_queue.put(None)
        tts_process.terminate()
        tts_process.join()
        sound_process.terminate()
        sound_process.join()
    else:
        console.print("\n[red]Exiting...")
        tts_process.close()
        tts_process.join()
        sound_process.close()
        sound_process.join()

    console.print("[blue]Session ended.")


if __name__ == "__main__":
    main()
