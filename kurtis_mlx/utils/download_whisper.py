import os
import argparse
import requests
from rich.console import Console
from rich.progress import Progress

console = Console()

MODELS = [
    "tiny",
    "tiny.en",
    "tiny-q5_1",
    "tiny.en-q5_1",
    "tiny-q8_0",
    "base",
    "base.en",
    "base-q5_1",
    "base.en-q5_1",
    "base-q8_0",
    "small",
    "small.en",
    "small.en-tdrz",
    "small-q5_1",
    "small.en-q5_1",
    "small-q8_0",
    "medium",
    "medium.en",
    "medium-q5_0",
    "medium.en-q5_0",
    "medium-q8_0",
    "large-v1",
    "large-v2",
    "large-v2-q5_0",
    "large-v2-q8_0",
    "large-v3",
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
]

SRC_DEFAULT = "https://huggingface.co/ggerganov/whisper.cpp"
PFX_DEFAULT = "resolve/main/ggml"

SRC_TDRZ = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"


def download_file(url, dest_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_length = int(response.headers.get("content-length", 0))

        with Progress() as progress:
            task = progress.add_task(
                f"[cyan]Downloading {os.path.basename(dest_path)}...",
                total=total_length,
            )
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download GGML Whisper models.")
    parser.add_argument("model", choices=MODELS, help="Model to download")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="models",
        help="Output directory (default: ./models)",
    )

    args = parser.parse_args()

    model = args.model
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    src = SRC_DEFAULT
    pfx = PFX_DEFAULT

    if "tdrz" in model:
        src = SRC_TDRZ
        pfx = PFX_DEFAULT  # Same prefix for this repo

    filename = f"ggml-{model}.bin"
    url = f"{src}/{pfx}-{model}.bin"
    dest_path = os.path.join(output_dir, filename)

    if os.path.exists(dest_path):
        console.print(
            f"[yellow]Model {model} already exists at {dest_path}. Skipping download.[/yellow]"
        )
        return

    try:
        console.print(f"Downloading model from {url}...")
        download_file(url, dest_path)
        console.print(f"[green]Successfully downloaded {model} to {dest_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]Failed to download model: {e}[/bold red]")


if __name__ == "__main__":
    main()
