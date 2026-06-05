"""
CUT MACHINE — RunPod Serverless Worker
VibeVoice-1.5B TTS — text → speech.

Input JSON:
{
  "input": {
    "text": "Hello world. This is a test.",
    "speaker_names": ["Alice"],          // optional, default ["Alice"]
                                          // available: Alice, Carter, Frank, Maya, Mary, Samuel, Anchen, Bowen, Xinran
    "cfg_scale": 1.3,                    // optional, default 1.3
    "ddpm_steps": 10                     // optional, default 10
  }
}

Output JSON:
{ "audio_b64": "...", "format": "wav", "sample_rate": 24000, "size_bytes": N }
"""
import os, sys, base64, io, traceback, tempfile
import runpod

# Boot diagnostics — appear in worker logs immediately
print("[boot] starting handler", flush=True)
print(f"[boot] python {sys.version}", flush=True)

import torch
print(f"[boot] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

# VibeVoice lives in /app/VibeVoice (cloned in Dockerfile, installed editable)
sys.path.insert(0, "/app/VibeVoice")

import numpy as np
import soundfile as sf
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

MODEL_ID = "vibevoice/VibeVoice-1.5B"
VOICES_DIR = "/app/VibeVoice/demo/voices"

print(f"[boot] loading processor from {MODEL_ID}", flush=True)
PROCESSOR = VibeVoiceProcessor.from_pretrained(MODEL_ID)

print(f"[boot] loading model {MODEL_ID} fp16 -> cuda", flush=True)
MODEL = VibeVoiceForConditionalGenerationInference.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa"
)
MODEL.eval()
print("[boot] model ready", flush=True)


def _voice_path(name):
    # Speaker voice samples shipped with the repo (e.g. en-Alice_woman.wav)
    for fname in os.listdir(VOICES_DIR):
        if name.lower() in fname.lower() and fname.endswith((".wav", ".mp3", ".flac")):
            return os.path.join(VOICES_DIR, fname)
    raise FileNotFoundError(f"no voice file found for '{name}' in {VOICES_DIR}")


def handler(job):
    try:
        inp = job["input"]
        text = inp["text"]
        speaker_names = inp.get("speaker_names") or ["Alice"]
        cfg_scale = float(inp.get("cfg_scale", 1.3))
        ddpm_steps = int(inp.get("ddpm_steps", 10))

        # Build conversation script in VibeVoice format ("Speaker 0: ..." for single voice).
        if isinstance(text, str) and not text.strip().startswith("Speaker"):
            text = f"Speaker 0: {text.strip()}"

        # Locate voice reference files.
        voice_paths = [_voice_path(n) for n in speaker_names]

        inputs = PROCESSOR(
            text=[text],
            voice_samples=[voice_paths],
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = MODEL.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=PROCESSOR.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
            )

        # outputs.speech_outputs is a list of torch tensors at 24kHz.
        wav = outputs.speech_outputs[0].cpu().float().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()

        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
        data = buf.getvalue()

        return {
            "audio_b64": base64.b64encode(data).decode("ascii"),
            "format": "wav",
            "sample_rate": 24000,
            "duration_sec": float(len(wav) / 24000),
            "size_bytes": len(data),
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
