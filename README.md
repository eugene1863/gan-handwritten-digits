# Handwritten Digit GAN API

A conditional GAN (cGAN) trained on MNIST that generates realistic handwritten
digit images, served through a FastAPI HTTP API. Conditioning on the digit
label (0-9) lets the API return a *specific* digit instead of a random one.

## Setup

```bash
pip install -r requirements.txt
```

## Train

Downloads MNIST automatically on first run and saves checkpoints as it goes.

```bash
python train.py --epochs 50 --batch-size 128
```

- Weights: `checkpoints/generator.pth` (used by the API)
- Sample grids per epoch: `samples/epoch_XXX.png` — check these to see training
  progress (digits should look increasingly realistic after ~15-20 epochs on
  GPU, longer on CPU).

## Run the API

```bash
uvicorn api:app --reload --port 8000
```

If `checkpoints/generator.pth` doesn't exist yet, the API still starts but
returns untrained noise — run `train.py` first for realistic digits.

### Endpoints

- `GET /health` — status and whether a trained checkpoint is loaded
- `GET /generate?digit=5` — single PNG image of the digit `5` (omit `digit`
  for a random digit)
- `GET /generate/grid?digit=5&count=16` — PNG grid of 16 samples of `5`
- `GET /generate/batch?digit=5&count=10` — JSON with base64-encoded PNGs

### Examples

```bash
curl "http://localhost:8000/generate?digit=7" --output digit7.png
curl "http://localhost:8000/generate/grid?digit=3&count=16" --output grid3.png
curl "http://localhost:8000/generate/batch?digit=9&count=5"
```

## Files

- `model.py` — Generator/Discriminator architectures (conditional DCGAN)
- `train.py` — training loop
- `api.py` — FastAPI service
