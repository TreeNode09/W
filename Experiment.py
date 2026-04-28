import os
import random
import json
import time
import io
from PIL import Image

from Interface import *

BASE_DIR = r"D:/W-Back/Data"
EXP_DIR = os.path.join(BASE_DIR, "exp")

PROMPTS_FILE = os.path.join(EXP_DIR, "prompts.json")

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
MODEL_ID = "sd-research/stable-diffusion-2-1-base"

def make_progress_logger(title: str):

    start_time = time.perf_counter()
    total = None

    def on_progress(current: int, total_count: int) -> None:

        nonlocal total

        if total is None:
            total = total_count

        elapsed = time.perf_counter() - start_time
        avg = elapsed / current
        it_per_sec = current / elapsed if elapsed > 0 else 0.0
        eta = (total - current) * avg

        print(
            f"[{title}] {current}/{total} | "
            f"elapsed={elapsed:.2f}s | avg={avg:.2f}s/img | "
            f"speed={it_per_sec:.2f} img/s | eta={eta:.2f}s"
        )

    return on_progress

def preparePrompts(num_prompts: int, seed: int = 42) -> list[str]:

    os.makedirs(EXP_DIR, exist_ok=True)
    random.seed(seed)
    return getPrompts(PROMPT_DATASET, num_prompts, EXP_DIR)


def generatePRCImages(num: int, watermark: bool, key_id: str | None = None) -> None:

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f: prompts = json.load(f)
    if watermark and not key_id: raise ValueError("key_id is required when watermark=True")

    out_dir_name = "prc" if watermark else "plain"
    out_path = os.path.join(EXP_DIR, out_dir_name)
    os.makedirs(out_path, exist_ok=True)

    applyPRC(BASE_DIR, key_id or "", MODEL_ID, prompts[:num], message_fn=lambda i: str(i), 
        out_path=out_path, watermark=watermark, on_progress=make_progress_logger(out_dir_name))

    return


def generateWaterLoImages(alpha: float, num: int, in_path: str) -> None:

    names = [n for n in os.listdir(in_path) if os.path.splitext(n)[1].lower() in {".png", ".jpg", ".jpeg"}]
    names = sorted(names, key=lambda n: int(os.path.splitext(n)[0]))
    images = [Image.open(os.path.join(in_path, n)).convert("RGB") for n in names[:num]]

    in_tag = os.path.basename(os.path.normpath(in_path))
    out_path = os.path.join(EXP_DIR, f"{in_tag}_wl_{alpha:g}")
    os.makedirs(out_path, exist_ok=True)

    applyWaterLo(images, BASE_DIR, alpha=alpha, out_path=out_path, on_progress=make_progress_logger(f"wl_{alpha:g}"))

    return


def detectPRCImages(key_id: str, num_images: int, images_dir: str, *, jpeg_quality: int | None = None) -> dict[str, float]:

    names = [n for n in os.listdir(images_dir) if os.path.splitext(n)[1].lower() in {".png", ".jpg", ".jpeg"}]
    names = sorted(names, key=lambda n: int(os.path.splitext(n)[0]))
    names = names[:num_images]
    images = [Image.open(os.path.join(images_dir, n)).convert("RGB") for n in names]

    if jpeg_quality is not None:

        images_jpeg: list[Image.Image] = []

        for im in images:

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            images_jpeg.append(Image.open(buf).convert("RGB"))
    
        images = images_jpeg

    results = decodePRC(BASE_DIR, key_id, MODEL_ID, images, on_progress=make_progress_logger("decode_prc"))

    detect_acc = sum(1 for detect, _ in results if detect) / len(results)

    decode_hits = 0
    decode_valid = 0

    for i, (_, bits) in enumerate(results):

        if bits is None: continue

        decoded_text = decodeBitsToText(bits)
        if decoded_text is None: continue

        decode_valid += 1
        if decoded_text == str(i): decode_hits += 1

    decode_acc_all = decode_hits / len(results)
    decode_acc_valid = (decode_hits / decode_valid) if decode_valid > 0 else 0.0

    metrics = {
        "detect_accuracy": detect_acc,
        "decode_accuracy_all": decode_acc_all,
        "decode_accuracy_valid": decode_acc_valid,
        "decode_valid_rate": (decode_valid / len(results)) if results else 0.0,
    }
    print(metrics)
    return metrics


if __name__ == "__main__":

    NUM_IMAGES = 500
    KEY_ID = "paste_key_id_here"
    ALPHAS = [0.002, 0.005, 0.01]

    prompts = preparePrompts(NUM_IMAGES, seed=42)
    print(f"Prepared {len(prompts)} prompt(s) at {PROMPTS_FILE}")

    # 1. no PRC, no WaterLo
    print("\n=== PLAIN IMAGE ===\n")
    generatePRCImages(NUM_IMAGES, watermark=False, key_id=None)

    # 2. PRC, no WaterLo
    print("\n=== PRC ===\n")
    generatePRCImages(NUM_IMAGES, watermark=True, key_id=KEY_ID)

    # 3. no PRC + WaterLo(alpha)
    for alpha in ALPHAS:

        print(f"\n=== WATERLO alpha={alpha} ===\n")
        generateWaterLoImages(alpha, NUM_IMAGES, os.path.join(EXP_DIR, "plain"))

    # 4. PRC + WaterLo(alpha)
    for alpha in ALPHAS: 
        
        print(f"\n=== PRC + WATERLO alpha={alpha} ===\n")
        generateWaterLoImages(alpha, NUM_IMAGES, os.path.join(EXP_DIR, "prc"))