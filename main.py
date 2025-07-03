from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from transformers import pipeline

app = FastAPI()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    result = asr("temp.wav")
    text = result["text"]
    emotion_scores = emotion_classifier(text)[0]
    # Найти эмоцию с максимальным score
    top_emotion = max(emotion_scores, key=lambda x: x["score"])
    return {
        "text": text,
        "emotion": top_emotion["label"],
        "emotion_score": top_emotion["score"],
        "all_emotions": emotion_scores
    }

@app.get("/transcribe-test/")
async def transcribe_test():
    test_wav_path = "test_data/test.wav"  # путь к вашему тестовому файлу
    result = asr(test_wav_path)
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