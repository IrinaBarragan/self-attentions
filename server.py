#stable-diffusion-v1-5 model
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline
import torch
import uvicorn
import os
from typing import Dict, List
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
from huggingface_hub import hf_hub_download, model_info

app = FastAPI()

# Configuration pour CPU/GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Enable memory-efficient attention
pipe.enable_xformers_memory_efficient_attention()  # Requires xformers in requirements.txt

# Enable model offloading
pipe.enable_model_cpu_offload()

# Enable sequential CPU offloading (most memory efficient)
pipe.enable_sequential_cpu_offload()
# Chargement du modèle
#MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
#MODEL_NAME ="segmind/SSD-1B"
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True
).to(device)

# Optimisations pour CPU
if device == "cpu":
    pipe.enable_attention_slicing()

class InputText(BaseModel):
    text: str

class TokenAnalysis(BaseModel):
    token: str
    importance: float

class AnalysisResponse(BaseModel):
    prompt: str
    tokens: List[TokenAnalysis]
    most_influential: str
    least_influential: str

def analyze_prompt(prompt: str) -> AnalysisResponse:
    # Tokenization
    tokenizer: CLIPTokenizer = pipe.tokenizer
    text_encoder: CLIPTextModel = pipe.text_encoder
    
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Forward pass through text encoder
    with torch.no_grad():
        text_embeddings = text_encoder(inputs.input_ids)[0]
    
    # Calcul de l'importance
    embeddings_norm = torch.norm(text_embeddings.squeeze(0), dim=1)
    importance_scores = embeddings_norm.cpu().numpy()
    
    # Filtrage des tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    valid_tokens = []
    valid_scores = []
    
    for token, score in zip(tokens, importance_scores):
        if token not in [tokenizer.pad_token, tokenizer.bos_token, tokenizer.eos_token]:
            valid_tokens.append(token)
            valid_scores.append(score)
    
    # Normalisation
    total = sum(valid_scores)
    normalized_scores = [score/total for score in valid_scores]
    
    # Tri par importance
    sorted_indices = np.argsort(normalized_scores)[::-1]
    
    return AnalysisResponse(
        prompt=prompt,
        tokens=[TokenAnalysis(token=t, importance=s) 
               for t, s in zip(valid_tokens, normalized_scores)],
        most_influential=valid_tokens[sorted_indices[0]],
        least_influential=valid_tokens[sorted_indices[-1]]
    )

@app.post("/attention")
async def analyze_text(request: InputText):
    return analyze_prompt(request.text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)