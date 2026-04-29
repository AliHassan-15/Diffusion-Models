## Diffusion Models (DDPM) — Assignment 04
From-scratch Denoising Diffusion Probabilistic Model (DDPM) in PyTorch for high-resolution face image generation and reconstruction.

### Course
Generative AI (AI4009) — Spring 2026

### Group members
- Ali Hassan (22F-3377)
- Bilal Nadeem (22F-3845)

---

## What this repo contains
- `AI_ASS04_DDPM.ipynb`: complete DDPM implementation (forward diffusion, U-Net denoiser, training, sampling, reconstruction, PSNR/SSIM, Gradio app).
- `Assignment04_Report.md`: detailed written report with figures.
- `Assignment04_Report.docx`: same report exported to Word.
- `_kaggle_imgs/`: saved output figures from Kaggle runs (loss curve, forward steps, reverse steps, generation, reconstruction, bonus comparisons).

---

## How to run (Kaggle)
1. Create a new Kaggle notebook and upload `AI_ASS04_DDPM.ipynb`.
2. Add dataset input: **CelebA-HQ-256 (images only)**.
3. Enable GPU (T4 is fine, T4×2 supported).
4. Run all cells.

Outputs (models/plots) are saved under `/kaggle/working`.

---

## Notes
- No pretrained diffusion pipelines are used.
- No HuggingFace diffusers are used.
- U-Net uses residual blocks + sinusoidal timestep embeddings, with a 64→128→256 channel progression.
- Bonus section includes DDIM sampling, cosine schedule comparison, and 256×256 sampling demonstration.

