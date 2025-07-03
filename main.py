from fastapi import FastAPI, UploadFile, File
from transformers import pipeline

app = FastAPI()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr("temp.wav")
    return {"text": result["text"]}