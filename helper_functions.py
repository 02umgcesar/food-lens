# helper_functions.py
import json


def load_json(name):
    """Load a file from the JSON/ folder, e.g. load_json("nutrition") -> JSON/nutrition.json"""
    with open(f"JSON/{name}.json") as f:
        return json.load(f)


def parse_class_name(name: str) -> str:
    """'  Apple  Pie ' -> 'apple_pie'"""
    trimmed = name.strip()                       # remove leading/trailing whitespace
    return "_".join(trimmed.lower().split())     # lowercase, spaces -> "_"
