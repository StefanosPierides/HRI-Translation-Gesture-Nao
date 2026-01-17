import socket
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import os
import time
import pygame
from openai import OpenAI

# ========================================================================
#   CONFIGURATION
# ========================================================================
PC_DEMO_MODE = False  # False = Robot Speaks. True = Laptop Speaks.

HOST = '127.0.0.1'
PORT = 8888
API_KEY = "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"

# LIST OF LANGUAGES NAO ACTUALLY HAS INSTALLED
SUPPORTED_NAO_LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Italian"]
# ========================================================================

FILENAME = "output.wav"
SPEECH_FILE = "speech.mp3"
SAMPLE_RATE = 44100

recording = False
current_role = None
audio_buffer = []
detected_language = None   # <-- FIXED (was Mandarin)
client_socket = None
gesture_tape = []
start_record_time = 0

pygame.mixer.init()


def audio_callback(indata, frames, time, status):
    if recording:
        audio_buffer.append(indata.copy())


def start_recording(role):
    global recording, audio_buffer, current_role, gesture_tape, start_record_time
    if not recording:
        current_role = role
        if role == "Foreign":
            mode_text = f"{detected_language} -> English"
            trigger_msg = "(Triggered by '2' / Victory)"
        else:
            mode_text = f"English -> {detected_language}"
            trigger_msg = "(Triggered by '1' / Pointing Up)"

        print(f"\n>>> RECORDING STARTED {trigger_msg}")
        print(f">>> Mode: {mode_text}")

        recording = True
        audio_buffer = []
        gesture_tape = []
        start_record_time = time.time()
        threading.Thread(target=run_audio_stream).start()
    else:
        print(f">>> Ignored Start Command (Already recording)")


def run_audio_stream():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while recording:
            sd.sleep(100)


def stop_and_process(stopper_role):
    global recording
    if recording:
        print(f">>> STOPPED. Processing...")
        recording = False

        if not audio_buffer:
            return

        recording_data = np.concatenate(audio_buffer, axis=0)
        max_volume = np.max(np.abs(recording_data))
        if max_volume < 0.005:
            print(">>> Silence detected. Ignoring.")
            return

        recording_int16 = (recording_data * 32767).astype(np.int16)
        wav.write(FILENAME, SAMPLE_RATE, recording_int16)

        process_smart_translation()


def replay_gestures():
    if not gesture_tape:
        return

    print(">>> [REPLAY]: Starting Gesture Replay on Robot...")
    try:
        replay_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        replay_socket.connect((HOST, PORT))

        first_move_time = 0
        found_move = False
        for record in gesture_tape:
            if "NONE" not in record['gesture']:
                first_move_time = record['time']
                found_move = True
                break

        if found_move:
            print(f">>> [REPLAY]: Trimming {first_move_time:.2f}s of initial stillness.")

        start_playback = time.time()

        for record in gesture_tape:
            if record['time'] < first_move_time:
                continue

            adjusted_rel_time = record['time'] - first_move_time
            target_time = start_playback + adjusted_rel_time
            sleep_duration = target_time - time.time()

            if sleep_duration > 0:
                time.sleep(sleep_duration)

            msg = f"FORCE:{record['gesture']}"
            replay_socket.sendall(msg.encode())

        replay_socket.close()
        print(">>> [REPLAY]: Finished.")
    except Exception as e:
        print(f"Replay Error: {e}")


def speak_text(text, target_lang="English"):
    print(f"\n[ROBOT ACTION REQ] Speaking: '{text}' (Voice: {target_lang})")

    if gesture_tape:
        threading.Thread(target=replay_gestures).start()

    if PC_DEMO_MODE:
        print(">>> [PC DEMO]: Playing audio on Laptop...")
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            time.sleep(0.1)
            client = OpenAI(api_key=API_KEY)
            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=text
            ) as response:
                response.stream_to_file(SPEECH_FILE)
            pygame.mixer.music.load(SPEECH_FILE)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.unload()
        except Exception as e:
            print(f"Error: {e}")

    else:
        print(">>> [NAO MODE]: Sending commands to Hub...")
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_sock.connect((HOST, PORT))

            # Select valid NAO voice
            nao_lang_setting = "English"
            for supported in SUPPORTED_NAO_LANGUAGES:
                if supported.lower() in target_lang.lower():
                    nao_lang_setting = supported
                    break

            # Send language
            lang_msg = f"LANG:{nao_lang_setting}"
            temp_sock.sendall(lang_msg.encode())
            time.sleep(0.5)

            # Send speech text
            clean_text = text.replace("\n", " ").strip()
            msg = f"SAY:{clean_text}"
            temp_sock.sendall(msg.encode())
            temp_sock.close()

        except Exception as e:
            print(f"Failed to send to robot: {e}")


def process_smart_translation():
    global detected_language
    try:
        client = OpenAI(api_key=API_KEY)

        # TRANSCRIBE
        with open(FILENAME, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        original_text = transcription.text.strip()
        if not original_text:
            return

        print(f"[{current_role} Input]: {original_text}")

        # -------------------------------------------------------------------
        # FOREIGN → ENGLISH MODE
        # -------------------------------------------------------------------
        if current_role == "Foreign":
            prompt = (
                f"User said: '{original_text}'. "
                "1. Identify language. "
                "2. Translate to English. "
                "Return: 'LANGUAGE: [Name] || TRANSLATION: [Text]'"
            )

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            result = response.choices[0].message.content

            try:
                parts = result.split("||")
                new_lang = parts[0].replace("LANGUAGE:", "").strip()
                translation = parts[1].replace("TRANSLATION:", "").strip()

                # Update detected language if needed
                if detected_language is None or (new_lang and "English" not in new_lang):
                    detected_language = new_lang

                speak_text(translation, "English")
            except:
                speak_text(result, "English")
            return

        # -------------------------------------------------------------------
        # ENGLISH → FOREIGN MODE
        # -------------------------------------------------------------------

        # SAFETY CHECK FIRST (fixes your NoneType bug)
        if detected_language is None:
            print(">>> No foreign language detected yet. Cannot translate English -> Foreign.")
            speak_text("I have not detected the foreign language yet.", "English")
            return

        # Check if NAO can speak this language
        is_supported = any(
            lang.lower() in detected_language.lower()
            for lang in SUPPORTED_NAO_LANGUAGES
        )

        # Supported language → Full native translation
        if is_supported:
            prompt = (
                f"Translate '{original_text}' into naturally spoken {detected_language}. "
                "Return ONLY text."
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            final_translation = response.choices[0].message.content
            speak_text(final_translation, detected_language)
            return

        # Unsupported language → PHONETIC MODE
        print(f">>> {detected_language} not installed. Using Phonetic Fallback.")

        prompt = (
            f"Translate '{original_text}' into {detected_language}. "
            f"Then convert that translation into English phonetics so an American robot "
            f"could read it and sound like it is speaking {detected_language}. "
            "Return ONLY the phonetic text."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        phonetic_text = response.choices[0].message.content

        print(f"[Phonetic Output]: {phonetic_text}")
        speak_text(phonetic_text, "English")

    except Exception as e:
        print(f"Error: {e}")


def main():
    global client_socket, gesture_tape
    print(f"Connecting to Hub at {HOST}:{PORT}...")
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("Connected! Controls: 1=English, 2=Foreign, Rock=Stop")

        last_trigger_time = 0
        debounce_seconds = 1.5

        while True:
            data = client_socket.recv(1024)
            if not data:
                break

            message = data.decode('utf-8')
            current_time = time.time()

            if recording:
                relative_time = current_time - start_record_time
                if "SAY:" not in message and "FORCE:" not in message:
                    gesture_tape.append({'time': relative_time, 'gesture': message})

            # Debounce triggers
            if current_time - last_trigger_time > debounce_seconds:
                if "ILoveYou" in message and recording:
                    stop_and_process(current_role)
                    last_trigger_time = current_time
                elif "Pointing_Up" in message and not recording:
                    start_recording("English")
                    last_trigger_time = current_time
                elif "Victory" in message and not recording:
                    start_recording("Foreign")
                    last_trigger_time = current_time

    except Exception as e:
        print(f"Connection Error: {e}")
    finally:
        if client_socket:
            client_socket.close()


if __name__ == "__main__":
    main()
