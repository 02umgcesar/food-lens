# main.py
import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from transformers import CLIPModel, CLIPProcessor

from helper_functions import load_json, parse_class_name


app = FastAPI()

# Load once at startup, not on every request
model = efficientnet_v2_s(num_classes=101)
model.load_state_dict(torch.load("effnetv2s_101_384 (91%).pt", map_location="cpu"))

class_names = load_json("class_names")  # Food-101 classes in torchvision's (alphabetical) order
nutrition = load_json("nutrition")      # nutrition per 100 g for each dish


model.eval()

# Filter 1: centroids (photo must look like one of the 101 dishes)
ood = torch.load("centroids.pt")
centroids = ood["centroids"]
SIM_THRESHOLD = ood["threshold"]  # 0.5535
assert ood["class_names"] == class_names and ood["input_size"] == 384
# Filter 2: model confidence
CONF_THRESHOLD = 0.5

# Filters 3 + 4: CLIP second opinion
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
dish_prompts = [f"a photo of {n.replace('_', ' ')}, a type of food." for n in class_names]
other_prompts = load_json("other_prompts")
with torch.no_grad():
    text = clip_processor(text=dish_prompts + other_prompts, return_tensors="pt", padding=True)
    clip_text = F.normalize(clip_model.get_text_features(**text).pooler_output, dim=-1)


# Must match test_transform used during training
preprocess = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    rejected = {"prediction": None, "message": "Not recognized"}
    
    model.eval()
    with torch.no_grad():
        features = torch.flatten(model.avgpool(model.features(preprocess(image).unsqueeze(0))), 1)
        probs = torch.softmax(model.classifier(features), dim=1)[0]
    pred = probs.argmax().item()

    # Filter 1: centroid similarity
    if (F.normalize(features, dim=1) @ centroids.T).max() < SIM_THRESHOLD:
        return rejected
    # Filter 2: confidence
    if probs[pred] < CONF_THRESHOLD:
        return rejected
    
    model.eval()
    with torch.no_grad():
        clip_img = F.normalize(clip_model.get_image_features(**clip_processor(images=image, return_tensors="pt")).pooler_output, dim=-1)
    scores = (clip_img @ clip_text.T)[0]
    dish_scores = scores[:len(class_names)]

    # Filter 3: CLIP thinks it's not a dish
    if scores.argmax() >= len(class_names):
        return rejected
    # Filter 4: CLIP disagrees (our answer not in CLIP's top 5)
    if pred not in dish_scores.topk(5).indices:
        return rejected

    dish = class_names[pred]
    n = nutrition[dish]
    return {"prediction": dish,
            "nutrition_per_100g": {"calories": n["calories"], "protein_g": n["protein"], "carbs_g": n["carbs"],
                                   "fat_g": n["fat"]}}


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/get-all-classes")
async def get_classes():
    return {"classes": class_names}


@app.get("/get-class/{class_name}")
async def get_class(class_name: str):
    parsed = parse_class_name(class_name)
    if parsed not in class_names:
        raise HTTPException(404, "Class not found")
    return {"class": parsed}