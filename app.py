import io
import os
import torch
import librosa
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModelForMultimodalLM, GenerationConfig
from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "google/gemma-4-12B-it"
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.70"))

processor: AutoProcessor = None
model: AutoModelForMultimodalLM = None
gen_config: GenerationConfig = None

def build_max_memory() -> dict:
    if not torch.cuda.is_available():
        return None
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(i).total_memory
        reserved_bytes = torch.cuda.memory_reserved(i)
        free_bytes = total_bytes - reserved_bytes
        allowed_bytes = int(free_bytes * GPU_MEMORY_UTILIZATION)
        max_memory[i] = allowed_bytes
    print(f"Using {max_memory} of memory.")
    return max_memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, gen_config
    max_memory = build_max_memory()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
        max_memory=max_memory,
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