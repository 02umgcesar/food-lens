# Food Lens 🍜

Upload a photo of a dish and Food Lens tells you **what it is** and its **nutrition per 100 g** (calories, protein, carbs, fat). It recognizes **101 dishes** from the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset, from pizza and ramen to tiramisu, and **says "Not recognized"** for photos that aren't one of them, like a dog, a chainsaw or a cup of coffee.

It runs locally as a FastAPI app with a simple web page.

## Features

- **Dish recognition:** a fine-tuned EfficientNetV2-S image classifier (101 classes)
- **Nutrition lookup:** calories, protein, carbs and fat per 100 g for every dish
- **Rejects photos it doesn't know:** four filters stop the model from forcing a label onto non-food photos or unfamiliar dishes
- **Browse and search:** a live search over all 100+ dishes, grouped alphabetically, so you can check whether a food is supported

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

A classifier trained on 100+ dishes always picks one of them, even for a photo of a dog. Each filter catches a different kind of mistake:

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


## Run it locally

Requires Python 3.12.

```bash
python3 -m pip install -r requirements.txt
fastapi dev main.py
```

Then open **http://127.0.0.1:8000**.

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
- **CLIP:** OpenAI, [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16)
- **Nutrition:** [USDA FoodData Central](https://fdc.nal.usda.gov/) (FNDDS, SR Legacy and Branded Foods; public domain) for 97 dishes. Carpaccio, croque madame, poutine and takoyaki use [Open Food Facts](https://world.openfoodfacts.org/) data, licensed under the ODbL.
