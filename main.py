# main.py
import threading
from user_input import start_user_input_thread
import audio_processing

if __name__ == "__main__":
    # Start the user inbput handler in a separate thread
    threading.Thread(target=start_user_input_thread, daemon=True).start()

    # Start the main audio processing loop
    audio_processing.process_audio()