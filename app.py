import io
import torch
import librosa
import numpy as np
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig

MODEL_ID = "google/gemma-4-12B-it"

processor: AutoProcessor = None
model: AutoModelForMultimodalLM = None
gen_config: GenerationConfig = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, gen_config
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    gen_config = GenerationConfig.from_pretrained(MODEL_ID)
    gen_config.max_new_tokens = 512
    yield
    del model
    del processor
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)

class TranscriptionResponse(BaseModel):
    transcription: str

@app.post("/gemma/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    language: str = Form(..., description="Language of the audio (e.g. 'English', 'Tamil', 'Hindi')"),
    audio: UploadFile = File(..., description="Audio file (wav, mp3, flac, ogg, etc.)"),
):
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an audio file.")

    raw_bytes = await audio.read()
    try:
        audio_array, _ = librosa.load(io.BytesIO(raw_bytes), sr=16000, mono=True)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio file: {e}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Transcribe the following speech segment in {language}. "
                        "Only output the transcription with no newlines, no commentary, and no explanation."
                    ),
                },
                {
                    "type": "audio",
                    "audio": audio_array,
                },
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, generation_config=gen_config)

    input_len = inputs["input_ids"].shape[-1]
    transcription = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()
    return TranscriptionResponse(transcription=transcription)