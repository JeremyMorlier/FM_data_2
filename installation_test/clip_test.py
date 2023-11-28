import torch
import clip
from PIL import Image

from thop import profile, clever_format

path = "third_party/CLIP/CLIP.png"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open(path)).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

print(model)
print(image.size())

macs, params = profile(model.encode_image, inputs=(image, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)

macs, params = profile(model.encode_text, inputs=(text, ))
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs :", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print("Expected    : [[0.9927937  0.00421068 0.00299572]] ")