from rich.console import Console
from kurtis_mlx.sip_client import SipClient

console = Console()


def sip_worker(
    transcription_queue, playback_queue, sip_server, sip_port, sip_user, sip_password
):
    """
    Manages the SIP client in a separate process.
    """
    try:
        queues = {"transcription": transcription_queue, "playback": playback_queue}

        # Pass the queues to the SipClient constructor
        sip_client = SipClient(
            server=sip_server,
            user=sip_user,
            password=sip_password,
            port=sip_port,
            queues=queues,
        )
        sip_client.run()

    except KeyboardInterrupt:
        console.print("\n[SIP Worker] Interrupted. Shutting down.")
    except Exception as e:
        console.print(
            f"[bold red][SIP Worker Error] An unexpected error occurred: {e}[/bold red]"
        )
    finally:
        console.print("[SIP Worker] Process finished.")
