import io
import os
import torch
import librosa
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration, GenerationConfig
from dotenv import load_dotenv
load_dotenv()

MODEL_ID = "google/gemma-4-12B-it"
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.70"))

processor: AutoProcessor = None
model: Gemma4UnifiedForConditionalGeneration = None
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
    print(f"[INFO] GPU max_memory map: {max_memory}")
    return max_memory


def load_audio(raw_bytes: bytes) -> np.ndarray:
    try:
        audio_array, _ = librosa.load(io.BytesIO(raw_bytes), sr=16000, mono=True)
        return audio_array.astype(np.float32)
    except Exception:
        pass

    try:
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(raw_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != 16000:
            audio_array = librosa.resample(
                audio_array.astype(np.float32), orig_sr=sr, target_sr=16000
            )
        return audio_array.astype(np.float32)
    except Exception:
        pass

    raise HTTPException(
        status_code=422,
        detail="Could not decode audio. Supported formats: wav, mp3, flac, ogg.",
    )

def build_prompt(language: str) -> str:
    lang_map = {
        "ta": ("Tamil", "தமிழ் எழுத்துக்கள்"),
        "hi": ("Hindi", "देवनागरी"),
        "en": ("English", "Latin script"),
    }
    lang_name, script_hint = lang_map.get(language, ("Tamil", "தமிழ் எழுத்துக்கள்"))
    return (
        f"Transcribe the following audio exactly as spoken in {lang_name}. "
        f"Output only the transcription in {script_hint}."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model, gen_config
    max_memory = build_max_memory()
    print(f"[INFO] Loading processor from {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[INFO] Loading model from {MODEL_ID} ...")
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    model.eval()
    gen_config = GenerationConfig.from_pretrained(MODEL_ID)
    gen_config.max_new_tokens = 1024
    gen_config.do_sample = False
    gen_config.repetition_penalty = 1.3
    print("[INFO] Model ready.")
    yield
    del model
    del processor
    torch.cuda.empty_cache()
    print("[INFO] Model unloaded.")


app = FastAPI(lifespan=lifespan, root_path="/inhouse_llm")

class TranscriptionResponse(BaseModel):
    text: str

@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(
    language: str = Form(default="ta"),
    file: UploadFile = File(...),
    response_format: str = Form(default="json"),
):

    await file.seek(0)
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")
 
    audio_array = load_audio(raw_bytes)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_prompt(language),
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
    return TranscriptionResponse(text=transcription)