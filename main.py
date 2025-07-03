import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import pipeline
import torch

app = FastAPI()

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device=device
)
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True,
    device=device
)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    try:
        # Сохраняем временный файл
        with open("temp_input.wav", "wb") as f:
            f.write(audio_bytes)
        # Преобразуем к 16kHz, моно
        y, sr = librosa.load("temp_input.wav", sr=16000, mono=True)
        sf.write("temp.wav", y, 16000)
        result = asr("temp.wav", return_timestamps=True)
        text = result["text"]
        try:
            # Ограничиваем длину текста для классификатора эмоций
            short_text = text[:512]
            emotion_scores = emotion_classifier(short_text)[0]
            top_emotion = max(emotion_scores, key=lambda x: x["score"])
            return {
                "text": text,
                "emotion": top_emotion["label"],
                "emotion_score": top_emotion["score"],
                "all_emotions": emotion_scores
            }
        except Exception as e:
            return {
                "text": text,
                "emotion_error": str(e)
            }
    except Exception as e:
        return {
            "error": str(e)
        }

@app.get("/transcribe-test/")
async def transcribe_test():
    # Преобразуем тестовый файл к нужному формату
    y, sr = librosa.load("test_data/test.wav", sr=16000, mono=True)
    sf.write("temp.wav", y, 16000)
    result = asr("temp.wav", return_timestamps=True)
    text = result["text"]
    emotion_scores = emotion_classifier(text)[0]
    top_emotion = max(emotion_scores, key=lambda x: x["score"])
    return {
        "text": text,
        "emotion": top_emotion["label"],
        "emotion_score": top_emotion["score"],
        "all_emotions": emotion_scores
    }

@app.get("/", response_class=HTMLResponse)
async def main_form():
    return """
    <html>
        <head>
            <title>Speech Recognition Upload</title>
        </head>
        <body>
            <h2>Загрузите WAV-файл для распознавания речи</h2>
            <form action="/transcribe/" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept=".wav">
                <input type="submit" value="Распознать">
            </form>
        </body>
    </html>
    """