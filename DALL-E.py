import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers import AutoencoderKL
from peft import LoraConfig, get_peft_model

# Determine device: prefer CUDA (GPU) if available, otherwise CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================
# 1. Load Pretrained CLIP Text Encoder
# =====================================
# We use OpenAI's CLIP text encoder to convert text prompts into dense
# vector embeddings. CLIP is kept frozen (no gradient updates) so that
# the semantic space remains stable while we train the diffusion prior
# and fine-tune the UNet via LoRA.

clip_model_name = "openai/clip-vit-large-patch14"
# Tokenizer converts raw text into token ids expected by the CLIP model.
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
# Load the CLIP text encoder and move it to the chosen device.
text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)
# Put the CLIP model into evaluation mode to disable dropout and other
# training-specific behaviors. We also disable gradients below.
text_encoder.eval()

# Freeze all parameters in the CLIP text encoder: we do not train CLIP.
for param in text_encoder.parameters():
    param.requires_grad = False


# =====================================
# 2. Load Stable Diffusion Components
# =====================================
# Load a pretrained Stable Diffusion pipeline which contains the VAE,
# UNet (denoiser), scheduler, and other helper utilities. We use the
# pipeline primarily to extract components (VAE, UNet) and the scheduler
# for training/inference operations.

model_id = "runwayml/stable-diffusion-v1-5"

# Load the pipeline; set `torch_dtype` to float16 for GPU memory savings
# when using CUDA. Move pipeline to the selected `device`.
pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

# Extract commonly used components from the pipeline for direct use.
vae = pipeline.vae            # Variational autoencoder (encoder/decoder)
unet = pipeline.unet          # Denoising UNet (used during diffusion)


# =====================================
# 3. Create Diffusion Prior Network
# =====================================
# The diffusion prior maps text embeddings (from CLIP) into an embedding
# space that the image generator condition expects (here treated simply
# as an MLP). In research implementations, the prior may be a diffusion
# model itself; here we use a small feed-forward network as a placeholder
# demonstrating the concept.

class DiffusionPrior(nn.Module):
    """
    Small MLP to map CLIP text embeddings to image-conditioned embeddings.

    Args:
        embed_dim (int): Dimensionality of the CLIP text embeddings.
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        # A 3-layer MLP with GELU activations projects the embedding to
        # a hidden space and back to the original dimensionality. This is
        # intentionally simple for clarity; larger or more complex priors
        # can be substituted.
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, text_embedding):
        # Forward pass projects the text embedding and returns the result.
        return self.net(text_embedding)


# Instantiate the prior and move it to the device.
prior = DiffusionPrior(embed_dim=768).to(device)


# =====================================
# 4. Apply LoRA to UNet for Fine-tuning
# =====================================
# LoRA (Low-Rank Adaptation) allows efficient fine-tuning by injecting
# small rank-factorized updates into selected modules of a large model.
# Here, we configure LoRA to target attention projection layers in the
# UNet (e.g., queries/keys/values). This drastically reduces trainable
# parameter count compared to full fine-tuning.

lora_config = LoraConfig(
    r=8,                             # rank of the LoRA update matrices
    lora_alpha=16,                   # scaling factor
    target_modules=["to_q", "to_k", "to_v"],  # modules to adapt
    lora_dropout=0.1,                # dropout applied to LoRA updates
    bias="none"                    # how to handle biases in LoRA layers
)

# Wrap the existing UNet with PEFT's LoRA adapter; this returns a model
# which behaves like the original UNet but has additional small, trainable
# LoRA parameters.
unet = get_peft_model(unet, lora_config)


# =====================================
# 5. Training Step (Simplified)
# =====================================
# This training step demonstrates the core operations needed to fine-tune
# the prior and LoRA-adapted UNet: tokenization, CLIP embedding, prior
# prediction, encoding images to latents, adding noise according to the
# scheduler, predicting the noise with UNet, and computing an MSE loss.
# This is a simplified example and omits dataset handling, batching,
# gradient accumulation, logging, checkpointing, and many production
# necessities.

optimizer = torch.optim.AdamW(
    list(prior.parameters()) + list(unet.parameters()),
    lr=1e-5
)


def training_step(images, captions):
    """
    Single training step for a batch of images and their captions.

    Args:
        images (torch.Tensor): Batch of images in pixel space (expects
            the format the VAE encoder expects, typically normalized).
        captions (List[str] or List[Union[str, ...]]): Corresponding text
            prompts for the images.

    Returns:
        float: Scalar training loss value.
    """

    # Tokenize the captions into the CLIP tokenizer's format. We set
    # padding/truncation to fit CLIP's expected max length (77 tokens).
    tokens = tokenizer(
        captions,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # Obtain CLIP text embeddings without computing gradients since CLIP
    # is frozen. We use the pooled representation (first token) here.
    with torch.no_grad():
        text_outputs = text_encoder(**tokens)
        # `last_hidden_state` shape: (batch, seq_len, hidden_dim).
        # We take the first token (CLIP uses a pooled token at index 0).
        text_embeddings = text_outputs.last_hidden_state[:, 0, :]

    # Feed text embeddings through the prior to obtain an image-style
    # embedding which we'll use as `encoder_hidden_states` to condition
    # the UNet (or alternatively as prompt embeddings when sampling).
    image_embeddings = prior(text_embeddings)

    # Encode input images into the VAE latent space. `encode` returns an
    # object with `latent_dist`; sampling from it gives latents for the
    # diffusion process. The scaling factor 0.18215 is commonly used in
    # Stable Diffusion implementations to match expected variance.
    latents = vae.encode(images).latent_dist.sample()
    latents = latents * 0.18215

    # Sample Gaussian noise and random timesteps for diffusion.
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

    # Add noise to latents according to the pipeline's scheduler.
    noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

    # Run the UNet to predict the noise given the noisy latents, the
    # timestep, and the conditioning embeddings. We expand `image_embeddings`
    # to match the UNet expected shape for `encoder_hidden_states`.
    noise_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=image_embeddings.unsqueeze(1)
    ).sample

    # Compute mean-squared-error between predicted and true noise.
    loss = nn.functional.mse_loss(noise_pred, noise)

    # Backpropagate and take an optimizer step. In a real training loop
    # you'd likely use gradient clipping, learning-rate scheduling, mixed
    # precision, and more robust optimizer state handling.
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


# Mark training completion for this step in the todo list


# =====================================
# 6. Inference
# =====================================
# Generate an image from a single prompt by obtaining CLIP embeddings,
# passing them through the prior to obtain conditioning embeddings, and
# then invoking the Stable Diffusion `pipeline` with `prompt_embeds`.

def generate(prompt):
    """
    Generate an image given a text `prompt` using the trained prior and
    the Stable Diffusion pipeline.

    Args:
        prompt (str): Text prompt describing the desired image.

    Returns:
        PIL.Image.Image: Generated image (first sample from the pipeline).
    """

    # Tokenize the single prompt and move tokens to the correct device.
    tokens = tokenizer(prompt, return_tensors="pt").to(device)

    # Compute CLIP text embedding and map it through the prior to obtain
    # an image embedding used as the pipeline's `prompt_embeds`.
    with torch.no_grad():
        text_outputs = text_encoder(**tokens)
        text_embedding = text_outputs.last_hidden_state[:, 0, :]

        # Map text embedding -> image embedding using the prior network.
        image_embedding = prior(text_embedding)

    # The pipeline expects prompt embeddings shaped appropriately. We
    # unsqueeze to add a sequence dimension if required.
    image = pipeline(
        prompt_embeds=image_embedding.unsqueeze(1)
    ).images[0]

    return image


# Example usage: generate an image from a short prompt. In practice,
# guard this call under `if __name__ == "__main__":` or use it from a
# script to avoid running on import.
if __name__ == "__main__":
    # Simple demo prompt (will run when executing the file directly).
    demo_image = generate("A futuristic city in watercolor style")
    # Optionally save or display the image; here we save to disk.
    demo_image.save("demo_futuristic_city.png")
