# 📦 Stable Diffusion from Scratch (Custom PyTorch Pipeline)
## 🌟 Overview
This project implements a Stable Diffusion-like pipeline from scratch using PyTorch. It allows you to generate realistic images from text prompts (text-to-image) and modify existing images using a reference prompt (image-to-image). The architecture closely mirrors Stable Diffusion 1.5, but is designed to be minimal, interpretable, and extendable.

## 🧠 What is Stable Diffusion?
Stable Diffusion is a type of latent diffusion model (LDM). It learns to denoise latent representations (compressed versions of images) instead of operating on pixel space, which makes it significantly faster and more memory-efficient than traditional pixel-space diffusion models.

## 🔧 Project Structure
```text
.
├── clip.py               # Text encoder (CLIP)
├── diffusion.py          # UNet-based denoiser model
├── encoder.py            # VAE encoder
├── decoder.py            # VAE decoder
├── ddpm.py               # DDPM noise scheduler and sampler
├── pipeline.py           # The main generation pipeline
├── model_loader.py       # Model loading utilities
├── model_converter.py    # Converts standard ckpt files into model state_dicts
├── data/
│   ├── v1-5-pruned.ckpt                # Pretrained full model weights
│   ├── v1-5-pruned-emaonly.ckpt       # EMA-only weights (smoother but less responsive)
│   ├── vocab                          # Tokenizer vocabulary (JSON)
│   ├── merges                         # BPE merge rules
├── demo.ipynb            # Notebook to run inference
└── README.md             # You are here.
```

## 🧩 Components and Their Roles
We'll now break down every component in detail.

## 1. 🔤 CLIP Text Encoder (`clip.py`)
### What is CLIP?
CLIP (Contrastive Language–Image Pre-training) is a model developed by OpenAI that can embed text and images into a shared latent space. It allows models to understand natural language prompts and associate them with visual content.

In our case, we use CLIP’s text encoder to turn a prompt (e.g., "a castle in the sky") into a dense vector (cond_context) that guides image generation.

How It Works in This Project
Uses a transformer-based architecture.

Tokenizes the prompt using a BPE tokenizer (vocab + merges from data/).

Outputs a tensor of shape (batch_size, seq_len, embedding_dim) used in cross-attention layers within the UNet.
```python
cond_tokens = tokenizer.batch_encode_plus([...]).input_ids
cond_context = clip(cond_tokens)
```
Where It Lives
File: `clip.py`

Loaded in `model_loader.py`

Used in: `pipeline.py` ```python (context = clip(tokens))```

## 2. 🎨 VAE: Encoder and Decoder (encoder.py, decoder.py)
### What is a VAE?
A Variational Autoencoder (`VAE`) compresses images into a smaller latent space and can reconstruct images from that space.

Encoder compresses a 512×512 RGB image into a (4, 64, 64) latent.

Decoder reconstructs the final image from the output of the diffusion process.

We work in latent space (4 × 64 × 64) to reduce computation cost and allow faster generation.

Encoder (VAE_Encoder)
Used during image-to-image (img2img) tasks.

Takes input image → compresses it to latent.

Adds noise during generation strength processing.

Decoder (VAE_Decoder)
Used during final step of text-to-image and img2img.

Takes final denoised latent and converts to RGB image.

Where They Live
Files: `encoder.py`, `decoder.py`

Loaded via: `model_loader.py`

Used in: `pipeline.py` ```python(encoder(), decoder())```

## 3. 🌀 Diffusion Model (diffusion.py)
### What is Diffusion?
Diffusion models iteratively denoise a noisy image until it becomes a clean image.

Here we implement a UNet architecture that predicts the noise added to latent images, given a timestep embedding and the prompt context.

UNet Details
Uses self-attention and cross-attention blocks.

Accepts:

x: latent noisy input

context: prompt embedding (from CLIP)

timestep: positional time embedding

```python
output = diffusion(latents, context, timestep_embedding)
```

Where It Lives
File: `diffusion.py`

Used in: `pipeline.py` (core denoising logic)

## 4. 🧊 DDPM Sampler (`ddpm.py`)
### What is DDPM?
#### Denoising Diffusion Probabilistic Models (DDPMs) define a forward and reverse diffusion process:
  - Forward Process: Gradually adds Gaussian noise to the data.
  - Reverse Process: A neural network learns to reverse this corruption step-by-step.

We use DDPM to sample from the latent space by:
  - Adding noise to an image latent
  - Predicting that noise using the UNet
  - Subtracting the prediction and updating over T steps

What It Does in This Project
  - `add_noise(latents, timestep)`: Adds forward noise to a latent at a given timestep.
  - `step(timestep, latents, model_output)`: Reverses noise using UNet prediction.

Where It Lives
File: `ddpm.py`

Used by: `pipeline.py`

You can replace this module to use alternate samplers (e.g., DDIM, Euler, LMS, Heun) later.

## 5. 🧪 The Generation Pipeline (pipeline.py)
This is the heart of the system — it coordinates everything to go from prompt and/or image → final image.

Supported Modes
| Mode             | What you provide         | What it does                                      |
| ---------------- | ------------------------ | ------------------------------------------------- |
| `text-to-image`  | `prompt` only            | Creates a new image based on text                 |
| `image-to-image` | `prompt` + `input_image` | Alters input image to match new prompt, using VAE |


## 🔄 Step-by-Step Breakdown
### 1. Tokenize Prompt
  - Use tokenizer with vocab + merges to tokenize the prompt (and uncond prompt for CFG).

### 2. Generate Text Embeddings
  - Forward tokens through clip() → Get prompt embedding (context).

### 3. Latent Preparation
  - If no image → sample Gaussian latents
  - If image given → encode via encoder() and inject noise based on strength

### 4. Sampling via DDPM
 #### 4.1 For each timestep:
  - Get positional embedding t
  - Predict noise: diffusion(latents, context, t)
  - Apply CFG (optional)
  - Update latent: latents = sampler.step(...)

### 5. Decode Final Latents
  - Decode with decoder() → RGB image
  - Rescale from [-1, 1] to [0, 255]

Return Image

```python
images = decoder(latents)
images = rescale(images, (-1, 1), (0, 255), clamp=True)
```

## 6. 🧳 Model Loader (model_loader.py)
###Responsible for:
  - Loading checkpoint file (.ckpt)
  - Extracting state dicts for:
    - clip
    - encoder
    - decoder
    - diffusion
  - Returning PyTorch-ready models on the correct device (CPU/CUDA/MPS)

This uses `model_converter.py` to extract and match submodules by key.

## 7. 🔄 Model Converter (model_converter.py)
This utility processes original `.ckpt` files (e.g., `v1-5-pruned.ckpt`) from HuggingFace or CompVis, and separates them into components:
```json
{
  'clip': {...},
  'encoder': {...},
  'decoder': {...},
  'diffusion': {...}
}
```
You can use:
```python 
state_dict = model_converter.load_from_standard_weights('v1-5-pruned.ckpt', device)
```

To make sure your `.ckpt` weights are loadable in your custom architecture.

## 8. 📁 `data/` Folder: Vocab, Merges, Checkpoints

| File Name                  | Description                                                               |
|---------------------------|---------------------------------------------------------------------------|
| `v1-5-pruned.ckpt`         | Full pretrained model weights (UNet, VAE, CLIP)                           |
| `v1-5-pruned-emaonly.ckpt` | EMA-smoothed weights for better quality but less responsive to fine edits |
| `vocab` (JSON)             | Dictionary mapping BPE tokens → IDs (used in tokenizer)                   |
| `merges` (Text)            | List of BPE merge rules to convert subwords into valid tokens             |


## 🔗 Downloads
If you are cloning the project, make sure to download the following files:
`https://drive.google.com/file/d/1_MwbvnqRY5pK39sGY3xpW0x32dZKdfAJ/view?usp=drive_link`


## ✅ Example Usage
You can generate images either via Python script or through the provided Jupyter notebook.
### 🖼 Text-to-Image Example

from pipeline import generate
```python
from model_loader import preload_models_from_standard_weights
import torch
```

# Setup
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = preload_models_from_standard_weights("data/v1-5-pruned.ckpt", device)

# Tokenizer setup (HuggingFace or custom BPE using vocab + merges)

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer(vocab_file="data/vocab", merges_file="data/merges")

# Generate image
image = generate(
    prompt="a futuristic city on Mars during sunset",
    do_cfg=True,
    cfg_scale=7.5,
    models=models,
    seed=42,
    device=device,
    tokenizer=tokenizer,
    n_inference_steps=50
)

# Save or show

from PIL import Image
Image.fromarray(image).save("output.png")
```

## 🖌 Image-to-Image Example
```python
from PIL import Image

input_image = Image.open("my_sketch.png")

image = generate(
    prompt="a photorealistic cat in a field of flowers",
    input_image=input_image,
    strength=0.7,
    do_cfg=True,
    cfg_scale=6.5,
    models=models,
    seed=1337,
    device=device,
    tokenizer=tokenizer,
    n_inference_steps=40
)
Image.fromarray(image).show()
```

## 🧠 Glossary of Core Concepts
| Term             | Meaning                                                               |
| ---------------- | --------------------------------------------------------------------- |
| **CLIP**         | Language model trained to match text and image embeddings             |
| **VAE**          | Variational Autoencoder used to compress images to 4×64×64 latents    |
| **Latent Space** | A compressed, abstract image representation used for efficiency       |
| **DDPM**         | Denoising Diffusion Probabilistic Model — learns how to reverse noise |
| **UNet**         | Architecture that denoises latent images using time + text conditions |
| **CFG**          | Classifier-Free Guidance: Boosts relevance to prompt                  |
| **Tokenization** | Turns prompt text into numerical IDs using `vocab` and `merges`       |
| **Seed**         | Determines random generator start point for reproducibility           |
| **Timesteps**    | Discrete denoising steps in reverse diffusion process                 |

## 🔧 File Breakdown for Pretrained Resources
| File                       | Format  | Purpose                                                   |
| -------------------------- | ------- | --------------------------------------------------------- |
| `v1-5-pruned.ckpt`         | `.ckpt` | Full model weights (UNet, VAE, CLIP)                      |
| `v1-5-pruned-emaonly.ckpt` | `.ckpt` | Exponential Moving Average weights — smoother, less noise |
| `vocab`                    | `.json` | Mapping of token → ID used in tokenizer                   |
| `merges`                   | `.txt`  | List of merge operations for byte pair encoding (BPE)     |

### 🔗 These files must be downloaded and placed in the /data/ folder. You can host them yourself or use HuggingFace links.

## 🚀 Extending the Project
This project is modular and easy to customize. You can:
  - 🔄 Swap out the sampler (DDIM, Euler, LMS, etc.)
  - 💡 Add attention mask control (e.g., token-level emphasis)
  - 🔌 Integrate LoRA, ControlNet, or textual inversion for fine-tuning
  - 🕸 Wrap in a web frontend (Streamlit, Gradio, or Flask API)
  - ⚙️ Export to ONNX or TorchScript for deployment

## 📝 License & Acknowledgements
This project is inspired by:
  - 🧠 Stable Diffusion by CompVis
  - 🤗 HuggingFace Transformers & Diffusers
  - 🖼 OpenAI’s CLIP
Models are for research and educational purposes only.
