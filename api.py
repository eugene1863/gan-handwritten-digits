"""FastAPI service that serves handwritten digit images from a trained
conditional GAN generator.

Run:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
    GET  /generate?digit=5                 -> single PNG image
    GET  /generate/grid?digit=5&count=16   -> PNG grid of N samples of that digit
    GET  /generate/batch?digit=5&count=10  -> JSON list of base64-encoded PNGs
"""
import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from torchvision.utils import make_grid
from PIL import Image

from model import Generator, LATENT_DIM, NUM_CLASSES

CHECKPOINT_PATH = os.environ.get("GENERATOR_CHECKPOINT", "checkpoints/generator.pth")

state = {"generator": None, "device": None}


def load_generator() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    if os.path.exists(CHECKPOINT_PATH):
        generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        # No trained weights yet: the model still runs, but output is
        # untrained noise rather than realistic digits.
        print(f"WARNING: checkpoint not found at {CHECKPOINT_PATH}. "
              "Run train.py first for realistic output.")
    generator.eval()
    state["generator"] = generator
    state["device"] = device


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_generator()
    yield


app = FastAPI(title="Handwritten Digit GAN API", lifespan=lifespan)


def _validate_digit(digit: Optional[int]) -> Optional[int]:
    if digit is not None and not (0 <= digit <= 9):
        raise HTTPException(status_code=400, detail="digit must be between 0 and 9")
    return digit


def _generate_images(digit: Optional[int], count: int) -> torch.Tensor:
    generator = state["generator"]
    device = state["device"]
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    with torch.no_grad():
        if digit is None:
            labels = torch.randint(0, NUM_CLASSES, (count,), device=device)
        else:
            labels = torch.full((count,), digit, dtype=torch.long, device=device)
        noise = torch.randn(count, LATENT_DIM, device=device)
        imgs = generator(noise, labels)  # range [-1, 1]
    return imgs


def _tensor_to_png_bytes(img_tensor: torch.Tensor) -> bytes:
    img = (img_tensor.clamp(-1, 1) + 1) / 2  # [0, 1]
    array = (img.squeeze(0).cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(array, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint_loaded": os.path.exists(CHECKPOINT_PATH),
        "device": str(state["device"]),
    }


@app.get("/generate")
def generate(digit: Optional[int] = Query(None, ge=0, le=9, description="Digit 0-9, omit for random")):
    _validate_digit(digit)
    imgs = _generate_images(digit, 1)
    png_bytes = _tensor_to_png_bytes(imgs[0])
    return Response(content=png_bytes, media_type="image/png")


@app.get("/generate/grid")
def generate_grid(
    digit: Optional[int] = Query(None, ge=0, le=9, description="Digit 0-9, omit for random"),
    count: int = Query(16, ge=1, le=64),
):
    _validate_digit(digit)
    imgs = _generate_images(digit, count)
    grid = make_grid(imgs, nrow=int(count ** 0.5) or 1, normalize=True)
    array = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/generate/batch")
def generate_batch(
    digit: Optional[int] = Query(None, ge=0, le=9, description="Digit 0-9, omit for random"),
    count: int = Query(10, ge=1, le=100),
):
    _validate_digit(digit)
    imgs = _generate_images(digit, count)
    images_b64 = [base64.b64encode(_tensor_to_png_bytes(img)).decode("utf-8") for img in imgs]
    return {"count": count, "digit": digit, "images_base64_png": images_b64}
