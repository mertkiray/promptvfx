import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import open_clip
from typing import List


# Function to load and preprocess images into a video tensor
def load_video(image_dir: str) -> torch.Tensor:
    transform = Compose([
        Resize((224, 224)),
        CenterCrop((224, 224)),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])

    image_tensors = []
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # Keep [0, 255] range
        image_tensor = transform(image_tensor / 255.0)  # Apply CLIP's normalization
        image_tensors.append(image_tensor)

    return torch.stack(image_tensors)  # Returns (frames, 3, 224, 224)


# Function to compute CLIP-based similarity scores
def clip_scores(video: torch.Tensor, prompt: List[str]):
    # Load CLIP model and tokenizer
    model = open_clip.create_model('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer(prompt)

    with torch.no_grad():
        video_features = model.encode_image(video)
        text_features = model.encode_text(text)
        video_features /= video_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity scores
        scores = 100.0 * video_features @ text_features.T
        naive_score = scores.mean().item()

        # Compute video frame similarity scores
        video_sim_score = 100.0 * video_features[:-1] @ video_features[1:].T
        advanced_score = video_sim_score.diag().mean().item()

    return naive_score, advanced_score


# Example usage
if __name__ == "__main__":
    image_dir = "render/example_explosion_1s/final/24"  # Change this to your actual image folder path
    video_tensor = load_video(image_dir)

    prompts = ["The vase explodes like a powder keg."]
    naive_score, advanced_score = clip_scores(video_tensor, prompts)

    print(f"Naive Score: {naive_score:.2f}")
    print(f"Advanced Score: {advanced_score:.2f}")