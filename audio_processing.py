# audio_processing.py
import numpy as np
import pyaudio
import config
import effects as eff

def process_audio():
    sample_rate = 96000  # Sample rate in Hz
    samples_per_buffer = 512
    frames_per_second = 60


    # Initialize PyAudio
    p = pyaudio.PyAudio()

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        print(f"Device {i}: {dev['name']}, Input Channels: {dev['maxInputChannels']}")


    stream_input = p.open(format=pyaudio.paFloat32,
                    input_device_index=1,
                    channels=p.get_device_info_by_index(1)['maxInputChannels'],
                    rate=sample_rate,
                    frames_per_buffer = samples_per_buffer,
                    input=True)

    stream_output = p.open(format=pyaudio.paFloat32,
                    channels=2,
                    rate=sample_rate,
                    frames_per_buffer = samples_per_buffer,
                    output=True)

    print("Audio processing started...")
    try:
        while True:
            # Read input audio
            input_chunk = stream_input.read(sample_rate//frames_per_second)
            # Apply the currently selected audio effect, if any
            chunk = eff.wave_former(input_chunk)

            # Output the processed audio chunk
            chunk = chunk[:, 0:2].astype(np.float32).flatten()
            stream_output.write(chunk.tobytes())

    except KeyboardInterrupt:
        print("Audio processing stopped.")
    finally:
        # Cleanup
        stream_output.stop_stream()
        stream_output.close()
        stream_input.stop_stream()
        stream_input.close()
        p.terminate()
