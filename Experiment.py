import os
import random
import json

from Interface import *

BASE_DIR = r"D:/W-Back/Data"
EXP_DIR = os.path.join(BASE_DIR, "exp")
PRC_IMG_DIR = os.path.join(EXP_DIR, "prc_images")
PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
MODEL_ID = "stabilityai/stable-diffusion-2-1-base"


def preparePrompts(num_prompts: int, seed: int = 42) -> list[str]:

    os.makedirs(EXP_DIR, exist_ok=True)
    random.seed(seed)
    return getPrompts(PROMPT_DATASET, num_prompts, EXP_DIR)


def generatePRCImages(key_id: str, *, prompts_path: str = PROMPTS_FILE) -> None:

    with open(prompts_path, "r", encoding="utf-8") as f: prompts = json.load(f)

    os.makedirs(PRC_IMG_DIR, exist_ok=True)
    applyPRC(EXP_DIR, key_id, MODEL_ID, prompts,
        message_fn=lambda i: i, out_path=PRC_IMG_DIR, watermark=True)

    return


if __name__ == "__main__":

    prompts = preparePrompts(1000)
    print(f"Prepared {len(prompts)} prompt(s) at {PROMPTS_FILE}")

    # generatePRCImages("")