import click
from rich.console import Console
from openai import OpenAI
from multiprocessing import Process, Queue as MPQueue

from kurtis_mlx import config
from kurtis_mlx.workers.tts import tts_worker
from kurtis_mlx.workers.sound import sd_worker
from kurtis_mlx.workers.sip import sip_worker
from kurtis_mlx.handlers import handle_interaction, handle_sip_interaction


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
    default="linroger023/Kurtis-E1.1-Qwen2.5-3B-Instruct-mlx-8Bit",
    help="LLM model identifier.",
)
@click.option(
    "--translate", is_flag=True, help="Translate assistant replies into user language."
)
@click.option(
    "--translation-model",
    default="ethicalabs/TowerInstruct-7B-v0.2-mlx-4Bit",
    help="Model to use for translation.",
)
@click.option("--sip", is_flag=True, help="Enable SIP/VoIP phone call mode.")
@click.option("--sip-server", help="SIP server (domain or IP).")
@click.option("--sip-port", default=5060, help="SIP server port.")
@click.option("--sip-user", help="SIP username.")
@click.option(
    "--sip-password",
    help="SIP password (or set SIP_PASSWORD env var).",
    envvar="SIP_PASSWORD",
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
    sip,
    sip_server,
    sip_port,
    sip_user,
    sip_password,
):
    if sip and not all([sip_server, sip_user, sip_password]):
        console.print(
            "[bold red]For SIP mode, you must provide --sip-server, --sip-user, and --sip-password.[/bold red]"
        )
        return

    history = [
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT.replace("phone call", "conversation"),
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
            samplerate if not sip else 8000,  # Use 8kHz for SIP
            lang_code,
            selected_speaker,
        ),
        daemon=True,
    )
    tts_process.start()

    # Start different audio worker based on mode
    if sip:
        transcription_queue = MPQueue()
        sip_process = Process(
            target=sip_worker,
            args=(
                transcription_queue,
                sound_queue,
                sip_server,
                sip_port,
                sip_user,
                sip_password,
            ),
            daemon=True,
        )
        sip_process.start()
    else:
        sound_process = Process(
            target=sd_worker,
            args=(sound_queue, samplerate),
            daemon=True,
        )
        sound_process.start()

    try:
        while True:
            if sip:
                # In SIP mode, we wait for audio from the sip_worker
                handle_sip_interaction(
                    text_queue,
                    transcription_queue,
                    full_whisper_model,
                    client,
                    history,
                    llm_model,
                    max_tokens,
                    translate,
                    language,
                    translation_model,
                )
            else:
                # In standard mode, we prompt for local microphone input
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
    finally:
        console.print("\n[blue]Shutting down workers...")
        text_queue.put(None)
        sound_queue.put(None)

        tts_process.join(timeout=5)
        if tts_process.is_alive():
            tts_process.terminate()

        if sip and "sip_process" in locals():
            sip_process.join(timeout=5)
            if sip_process.is_alive():
                sip_process.terminate()
        elif not sip and "sound_process" in locals():
            sound_process.join(timeout=5)
            if sound_process.is_alive():
                sound_process.terminate()

    console.print("[blue]Session ended.")


if __name__ == "__main__":
    main()
