import click
from rich.console import Console
from openai import OpenAI
from multiprocessing import Process, Queue as MPQueue, Event

from kurtis_mlx import config
from kurtis_mlx.workers.tts import tts_worker
from kurtis_mlx.workers.sound import sd_worker
from kurtis_mlx.workers.sip import sip_worker
from kurtis_mlx.workers.mic import mic_worker
from kurtis_mlx.handlers import handle_interaction, handle_sip_interaction
from kurtis_mlx.utils.tts import text_to_speech
from kurtis_mlx.ui import run_ui


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
    help="Base Whisper model (combined with language code) or OpenAI model name.",
)
@click.option(
    "--whisper-model-openai",
    default="Whisper-Large-v3-Turbo1",
    help="OpenAI-compatible Whisper model name (used with --whisper-backend openai).",
)
@click.option(
    "--whisper-backend",
    type=click.Choice(["mlx", "whisper_cpp", "openai"]),
    default="mlx",
    help="Backend for Whisper transcription.",
)
@click.option(
    "--ggml-model-path",
    help="Path to GGML model file for whisper-cpp-python.",
)
@click.option(
    "--tts-model",
    default="tts_models/multilingual/multi-dataset/xtts_v2",
    help="TTS model subpath",
)
@click.option("--max-tokens", default=1024, help="Maximum tokens in LLM response.")
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
    default="ethicalabs/Tower-Plus-2B-mlx",
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
@click.option(
    "--assistant-prompt",
    help="Initial assistant greeting. Assistant will say this and wait for user.",
)
@click.option("--ui", is_flag=True, help="Enable the graphical UI.")
@click.option("--debug", is_flag=True, help="Enable verbose debug logging.")
def main(
    language,
    speaker,
    whisper_model,
    whisper_model_openai,
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
    assistant_prompt,
    ui,
    whisper_backend,
    ggml_model_path,
    debug,
):
    # Set global LLM model from CLI arg
    config.LLM_MODEL = llm_model

    # Queue for sending audio RMS to UI
    ui_queue = MPQueue() if ui else None

    # Queue for sending control messages from UI to workers
    control_queue = MPQueue() if ui else None

    # Event to signal when audio is playing (to pause recording)
    is_busy_event = Event()
    interrupt_event = Event()
    pause_event = Event()

    # CLI Pause Listener
    if not ui:
        from pynput import keyboard

        def on_press(key):
            if key == keyboard.Key.space:
                if pause_event.is_set():
                    pause_event.clear()
                    console.print("[bold green]Resumed.[/bold green]")
                else:
                    pause_event.set()
                    console.print("[bold yellow]Paused.[/bold yellow]")

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

    def start_agent_logic():
        if sip and not all([sip_server, sip_user, sip_password]):
            console.print(
                "[bold red]For SIP mode, you must provide --sip-server, --sip-user, and --sip-password.[/bold red]"
            )
            return

        # Set initial LLM model from CLI arg if not already set (e.g. by UI settings loading?)
        # Actually, UI settings are loaded in run_ui, but start_agent_logic is passed to it.
        # If we want CLI to override settings, we should set it here.
        # But if we want settings to persist, we should check if it's already set?
        # For now, let's say CLI overrides initial, but UI can change it at runtime.
        if config.LLM_MODEL is None:
            config.LLM_MODEL = llm_model

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

        text_queue = MPQueue()
        sound_queue = MPQueue()
        transcription_queue = MPQueue()

        tts_process = Process(
            target=tts_worker,
            args=(
                text_queue,
                sound_queue,
                full_tts_model,
                samplerate if not sip else 8000,  # Use 8kHz for SIP
                lang_code,
                selected_speaker,
                interrupt_event,
            ),
            daemon=True,
        )
        tts_process.start()

        # Assistant starts with a greeting
        if assistant_prompt and not sip:
            console.print(f"[cyan]Assistant (Initial): {assistant_prompt}")
            # Add to history so the LLM knows it said this
            # TODO: add to history also for SIP call
            history.append({"role": "assistant", "content": assistant_prompt})

        # Start different audio worker based on mode
        if sip:
            if assistant_prompt:
                assistant_prompt_au = text_to_speech(
                    full_tts_model,
                    lang_code,
                    speaker,
                    22050,
                    8000,
                    assistant_prompt,
                )
            else:
                assistant_prompt_au = None
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
                    interrupt_event,
                    assistant_prompt_au,
                ),
                daemon=True,
            )
            sip_process.start()
        else:
            sound_process = Process(
                target=sd_worker,
                args=(
                    sound_queue,
                    samplerate,
                    is_busy_event,
                    interrupt_event,
                    pause_event,
                    ui_queue,
                ),
                daemon=True,
            )
            sound_process.start()
            mic_process = Process(
                target=mic_worker,
                args=(
                    transcription_queue,
                    is_busy_event,
                    pause_event,
                    ui_queue,
                    control_queue,
                    debug,
                ),
                daemon=True,
            )
            mic_process.start()

        try:
            while True:
                # Re-instantiate client to pick up any config changes (e.g. API URL from UI)
                client = OpenAI(
                    base_url=config.OPENAI_API_URL, api_key=config.OPENAI_API_KEY
                )

                if sip:
                    # In SIP mode, we wait for audio from the sip_worker
                    handle_sip_interaction(
                        text_queue,
                        sound_queue,
                        transcription_queue,
                        full_whisper_model,
                        client,
                        history,
                        config.LLM_MODEL,
                        max_tokens,
                        translate,
                        language,
                        translation_model,
                        interrupt_event,
                        whisper_backend=whisper_backend,
                        ggml_model_path=ggml_model_path,
                        whisper_model_openai=whisper_model_openai,
                    )
                else:
                    # In standard mode, we wait for local microphone input
                    handle_interaction(
                        text_queue,
                        sound_queue,
                        transcription_queue,
                        full_whisper_model,
                        client,
                        history,
                        config.LLM_MODEL,
                        max_tokens,
                        translate,
                        language,
                        translation_model,
                        is_busy_event,
                        interrupt_event,
                        whisper_backend=whisper_backend,
                        ggml_model_path=ggml_model_path,
                        whisper_model_openai=whisper_model_openai,
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
            elif not sip:
                if "sound_process" in locals():
                    sound_process.join(timeout=5)
                    if sound_process.is_alive():
                        sound_process.terminate()
                if "mic_process" in locals():
                    mic_process.join(timeout=5)
                    if mic_process.is_alive():
                        mic_process.terminate()

        console.print("[blue]Session ended.")

    if ui and sip:
        console.print("[bold red]UI mode is not compatible with SIP mode.[/bold red]")
        return

    if ui:
        run_ui(
            agent_run_function=start_agent_logic,
            audio_queue=ui_queue,
            control_queue=control_queue,
            debug=debug,
        )
    else:
        start_agent_logic()


if __name__ == "__main__":
    main()
