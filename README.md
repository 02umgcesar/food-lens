# Food Lens 🍜

Upload a photo of a dish and Food Lens tells you **what it is** and its **nutrition per 100 g** (calories, protein, carbs, fat). It recognizes **101 dishes** from the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset, from pizza and ramen to tiramisu, and **says "Not recognized"** for photos that aren't one of them, like a dog, a chainsaw or a cup of coffee.

It runs locally as a FastAPI app with a simple web page.

## Features

- **Dish recognition:** a fine-tuned EfficientNetV2-S image classifier (101 classes, **91.4% test accuracy** on the full Food-101 test set)
- **Nutrition lookup:** calories, protein, carbs and fat per 100 g for every dish
- **Rejects photos it doesn't know:** four filters stop the model from forcing a label onto non-food photos or unfamiliar dishes
- **Browse and search:** a live search over all 101 dishes, grouped alphabetically, so you can check whether a food is supported

## How it works

```
photo ─► EfficientNetV2-S ─► fingerprint (1280 features) + dish probabilities
           │
           ├─ 1. Centroid check     Is the photo's fingerprint close to any dish's average fingerprint?
           ├─ 2. Confidence check   Is the model at least 50% sure of its top dish?
           ├─ 3. CLIP "other" check Does CLIP think a generic label (fruit, drink, object, animal…) fits better than any dish?
           └─ 4. CLIP agreement     Is the model's answer among CLIP's top-5 dishes?
           │
           ▼
   dish + nutrition  —or—  "Not recognized"
```

A classifier trained on 101 dishes always picks one of them, even for a photo of a dog. An early version of this model labeled a photo of **ice cubes as "oysters" with 87% confidence**. Each filter catches a different kind of mistake:

- The **centroid check** compares the photo to the average "fingerprint" of each dish's training photos.
- The **confidence check** drops answers the model is unsure about.
- **CLIP** (OpenAI's general-purpose image–text model) adds a second opinion from a model that knows what apples, golf balls and coffee look like.

## Results

These were measured on 505 Food-101 test photos (5 per dish) and 398 photos of things the model was never trained on.

| Metric | Result |
|---|---|
| Top-1 accuracy (all 505 real dish photos) | **89.7%** |
| Correct dish in top 5 | 98.2% |
| Real dish photos answered (passed all filters) | 93.1% |
| Accuracy on answered photos | **93.8%** |
| Non-food photos rejected (animals, objects, places) | **100%** (150/150) |
| Ingredients and drinks rejected (apples, broccoli, coffee…) | 90% |
| Dishes outside the 101 rejected (bagel, croissant, kimchi…) | 61% |
| **Unseen photos handled sensibly** (rejected, or mapped to a near-identical dish like burrito → breakfast burrito) | **89.7%** |

The classifier on its own, without the filters, reaches **91.4% top-1 accuracy** on the full Food-101 test set (25,250 images).

## How the model was trained

I worked up to the final model step by step, starting with a small CNN built from scratch and moving to transfer learning. The first experiments used a 10-dish subset (7,500 training and 2,500 test images) to iterate quickly. The last ones used all 101 dishes.

| Model | Dishes | Image size | Training | Test accuracy |
|---|---|---|---|---|
| TinyVGG | 10 | 64×64 | From scratch, with augmentation | ~42% |
| ResNet-18 | 10 | 64×64 | From scratch, with weight decay | ~60% |
| ResNet-18 | 10 | 224×224 | Pretrained (ImageNet), fine-tuned, with augmentation | ~93% |
| EfficientNet-B2 | 10 | 224×224 | Pretrained, fine-tuned | ~96% |
| EfficientNet-B2 | 101 | 224×224 | Pretrained, fine-tuned | ~84% |
| EfficientNet-B2 | 101 | 288×288 | Pretrained, fine-tuned | 87.4% |
| **EfficientNetV2-S** | **101** | **384×384** | **Pretrained, fine-tuned** | **91.4%** |

What made the biggest difference:

- **Transfer learning:** starting from ImageNet weights instead of random ones took the 10-dish task from ~60% to over 90%.
- **Fine-tuning every layer:** training only the new classifier on top of a frozen ResNet-18 plateaued around 85%. Unfreezing everything with a small learning rate pushed it past 90%.
- **Data augmentation:** `TrivialAugmentWide` closed most of the gap between training and test accuracy.
- **Using each model's native resolution:** EfficientNet-B2 went from 84% to 87.4% on all 101 dishes just by moving from 224×224 to its intended 288×288.
- **A stronger pretrained model:** EfficientNetV2-S at 384×384 added another 4 percentage points.

**Final model setup:** EfficientNetV2-S with ImageNet weights from torchvision, and a new classifier (dropout 0.3, then 1280 → 101). All layers were fine-tuned for 7 epochs at 384×384 with `TrivialAugmentWide`, AdamW (learning rate 1e-4), label smoothing 0.1 and a cosine learning-rate schedule. Training took about 5.5 hours on a free Kaggle GPU.

## Limitations

- **Similar-looking dishes get mixed up.** For example, ramen and pho (both noodle soups in a bowl), or hamburger and steak.
- **Only 101 dishes.** About 4 in 10 photos of other dishes (bagel, croissant, kimchi…) still get labeled as a similar known dish instead of being rejected.
- **Nutrition is an average per 100 g** for each dish type, not an estimate for the specific plate or portion in the photo.
- **Food-101's training labels are noisy.** The dataset authors deliberately left some mislabeled training images in.

## Project structure

```
food-lens/
├── main.py                        # FastAPI app: model loading, the four filters and the endpoints
├── helper_functions.py            # JSON loading and dish-name parsing
├── index.html                     # Web page (upload, results and dish search)
├── effnetv2s_101_384 (91%).pt     # Fine-tuned EfficientNetV2-S weights (~82 MB)
├── centroids.pt                   # Average feature "fingerprint" per dish and the similarity threshold
├── requirements.txt
└── JSON/
    ├── class_names.json           # The 101 dishes, in torchvision's order
    ├── nutrition.json             # Nutrition per 100 g for each dish
    └── other_prompts.json         # CLIP's non-dish labels (fruit, drink, animal…)
```

## Run it locally

Requires Python 3.12.

```bash
python3 -m pip install -r requirements.txt
fastapi dev main.py
```

Then open **http://127.0.0.1:8000**. The first start downloads the CLIP model (about 600 MB) from Hugging Face.

## API

| Endpoint | Description |
|---|---|
| `GET /` | The web page |
| `POST /predict` | Upload an image and get `{"prediction": "ramen", "nutrition_per_100g": {...}}` or `{"prediction": null, "message": "Not recognized"}` |
| `GET /get-all-classes` | List of all 101 supported dishes |
| `GET /get-class/{name}` | Check whether a dish is supported (e.g. `/get-class/apple pie` → `apple_pie`) |

Interactive API docs are at **http://127.0.0.1:8000/docs**.

## Data and credits

- **Food-101:** Bossard, Guillaumin & Van Gool, *Food-101: Mining Discriminative Components with Random Forests*, ECCV 2014
- **EfficientNetV2-S:** Tan & Le, *EfficientNetV2: Smaller Models and Faster Training*, ICML 2021; pretrained weights from [torchvision](https://pytorch.org/vision/stable/models.html)
- **CLIP:** OpenAI, [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16)
- **Nutrition:** [USDA FoodData Central](https://fdc.nal.usda.gov/) (FNDDS, SR Legacy and Branded Foods; public domain) for 97 dishes. Carpaccio, croque madame, poutine and takoyaki use [Open Food Facts](https://world.openfoodfacts.org/) data, licensed under the ODbL.
