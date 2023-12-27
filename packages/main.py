import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import time
from openai import OpenAI
import numpy as np
import librosa
import pygame


api_key = 'LETSUSEDOTENV'
client = OpenAI(api_key=api_key)

fs = 44100
seconds = 0.5
def new_record_audio(recordings):
    # to record audio as wav file
    #print("Recording... Press 's' to stop.")

    print("Recording...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished

    # Convert stereo to mono if needed
    audio_mono = np.mean(recording, axis=1)

    # Use librosa to check if there's significant audio (speech)
    rms = librosa.feature.rms(y=audio_mono).mean()
    if rms > 0.01:  # Adjust the threshold as needed
        new_recordings = np.concatenate((recordings, recording), axis=0)
        return new_record_audio(recordings =  new_recordings)
    else:
        return recordings

def mix_recordings(recordings):
    audio_name = 'output'
    write(f'./{audio_name}.wav', fs, recordings)  # Save as WAV file 
    return f'./{audio_name}.wav'

def transcribe_audio(path, local=False):
    if(local):
        model = whisper.load_model("base")
        result = model.transcribe(path)
        return result["text"]
    else: 
        audio_file= open(path, "rb")
        transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        language="sv",
        file=audio_file)
        print('Transcript:', transcript)
        return transcript.text

def run_assistant(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},

        ]
    )
    answer_message = response.choices[0].message
    print('Assistant:', response.choices[0].message.content) # Här kan man lägga till func. call och så
    return answer_message.content

def play_mp3(file_path):
    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    
    # Wait for the playback to finish
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(1)
        time.sleep(0.1)

def create_speech(answer):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=answer,
    )
    file_name = "recording.mp3"
    response.stream_to_file(file_name)
    play_mp3(file_name)

def main(): 
    while True:
        recordings = np.array([]).reshape(0,2)
        recordings = new_record_audio(recordings)
        print('New recording detected  of length', recordings.shape)
        if recordings.shape[0]>0:
            path = mix_recordings(recordings)
            transcript = transcribe_audio(path)
            answer = run_assistant(transcript)
            create_speech(answer)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
