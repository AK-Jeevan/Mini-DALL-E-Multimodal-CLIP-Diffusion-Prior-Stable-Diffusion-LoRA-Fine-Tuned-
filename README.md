# 🎨 Mini DALL·E Multimodal (CLIP + Diffusion Prior + Stable Diffusion + LoRA)

A minimal research-style implementation of a **DALL·E-inspired multimodal text-to-image system** built with:

- **CLIP (ViT-L/14)** for text embeddings  
- A custom **Diffusion Prior (MLP)**  
- **Stable Diffusion v1.5** components (VAE + UNet + Scheduler)  
- **LoRA fine-tuning** for efficient UNet adaptation  
- End-to-end training pipeline  

This project demonstrates how modern text-to-image systems connect semantic embeddings to diffusion-based image generation in a simplified, educational format.

---

## 🧠 Architecture Overview

### 1️⃣ CLIP Text Encoder  
Uses OpenAI CLIP (ViT-L/14) to convert text prompts into semantic embeddings.  
CLIP is **frozen** during training to maintain a stable semantic space.

### 2️⃣ Diffusion Prior (MLP)  
A lightweight 3-layer MLP that maps CLIP embeddings into the conditioning space expected by the diffusion model.

### 3️⃣ Stable Diffusion Components  
Extracted from Stable Diffusion v1.5:
- **VAE** → Encodes images into latent space  
- **UNet** → Predicts noise during diffusion  
- **Scheduler** → Adds/removes noise  

### 4️⃣ LoRA Fine-Tuning  
Applies **Low-Rank Adaptation (LoRA)** to the UNet attention layers:
- Targets: `to_q`, `to_k`, `to_v`
- Rank: 8  
- Alpha: 16  
- Dropout: 0.1  

This drastically reduces trainable parameters compared to full fine-tuning.

---

## 🚀 Training (Simplified Step)

The training step:

1. Tokenizes captions  
2. Generates CLIP embeddings  
3. Maps embeddings through Diffusion Prior  
4. Encodes images with VAE  
5. Adds noise via scheduler  
6. Predicts noise with LoRA-adapted UNet  
7. Computes MSE loss  
8. Backpropagates & updates weights  

---

## 🔬 Key Concepts Demonstrated

- Multimodal alignment (text → image latent space)  
- Diffusion training objective (noise prediction)  
- Latent diffusion models  
- LoRA parameter-efficient fine-tuning  
- Freezing large pretrained encoders  
- Conditioning via learned embedding projection  

---

## 📌 Notes

- This is a **minimal educational implementation**, not a full production DALL·E replica.  
- The Diffusion Prior is simplified (MLP instead of a full diffusion prior model).  
- Dataset loading, batching, logging, and mixed precision are omitted for clarity.  
- Stable Diffusion weights are required and downloaded automatically from HuggingFace.  

---

## 🧩 Future Improvements

- Replace MLP prior with a diffusion prior  
- Add dataset + dataloader  
- Mixed precision training  
- Gradient accumulation  
- Configurable training loop  
- Checkpoint saving/loading  

---

## 📜 License

For research and educational purposes.  
Please respect the licenses of:
- OpenAI CLIP  
- Stable Diffusion  
- HuggingFace Diffusers  

---

If you found this helpful, consider ⭐ starring the repository!
