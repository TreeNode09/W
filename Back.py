import base64
import io
import math
import os
from typing import Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from jsonschema import ValidationError, validate

from Interface import *

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

BASE_DIR = r"D:\W\Data"
WATERLO_DIR = os.path.join(BASE_DIR, "waterlo")

PROMPT_DATASET = "Gustavosta/Stable-Diffusion-Prompts"
BATCH_SIZE = 8

def _json_dict() -> dict[str, Any] | None:

    d = request.get_json(silent=True)

    return d if isinstance(d, dict) else None


def _png_to_b64(img: Any) -> str:

    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


@app.route("/key", methods=["GET"])
def handle_key():
    """Generate a PRC key.
    
    ## Return:
    - key_id of the generated key
    """

    try: return jsonify({"key_id": generateKey(BASE_DIR)})

    except ValueError as e: return jsonify({"error": str(e)}), 400


@app.route("/prompts", methods=["GET"])
def handle_prompts():
    """Get `num` promtps from the given prompt dataset.
    
    ## Query:
    - `num`: positive `int`
    - (optional) `dataset_id`: non-empty `string`

    ## Return:
    - list of fetched prompts, length = `num`
    """

    SCHEMA_PROMPTS: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "num": {"type": "string", "pattern": "^[1-9][0-9]*$"},
            "dataset_id": {"type": "string", "minLength": 1}
        },
        "required": ["num"]
    }

    raw = {k: request.args[k] for k in request.args}

    try: validate(instance=raw, schema=SCHEMA_PROMPTS)
    except ValidationError as e: return jsonify({"error": e.message}), 400

    num = int(raw["num"])
    dataset_id = raw.get("dataset_id", PROMPT_DATASET).strip()

    try: prompts = getPrompts(dataset_id, num)
    except ValueError as e: return jsonify({"error": str(e)}), 400

    return jsonify({"prompts": prompts, "count": len(prompts)})


@app.route("/generate/prompts", methods=["POST"])
def handle_generate_by_prompts():
    """Generate images with `prompts` and a model with `model_id`, and apply selected watermarks.

    ## Body:
    - `model_id`: non-empty `string`
    - `prompts`: non-empty `array` of `string`
    - `use_prc`: `boolean`
    - `use_waterlo`: `boolean`
    - `key_id`:
        - if `use_prc` is `True`: non-empty `string`
        - if `use_prc` is `False`: ignored
    - (optional) `alpha`: `float` between (0, 1]
    
    ## Return: 
    - generated base-64 PNGs, length = `len(prompts)`

    ## Socket Event:
    - `generate_progress`: `{"current": int, "total": int, "percent": float}`
    """
    data = _json_dict()
    if data is None: return jsonify({"error": "JSON object body required"}), 400

    merged = dict(data)
    merged.setdefault("alpha", 0.005)

    SCHEMA_GENERATE_BY_PROMTPS: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["model_id", "prompts", "use_prc", "use_waterlo", "alpha"],
        "properties": {
            "model_id": {"type": "string", "minLength": 1},
            "prompts": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "use_prc": {"type": "boolean"},
            "use_waterlo": {"type": "boolean"},
            "key_id": {"type": ["string", "null"]},
            "alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}
        },
        "allOf": [
            {
                "if": {"properties": {"use_prc": {"const": True}}},
                "then": {
                    "required": ["key_id"],
                    "properties": {"key_id": {"type": "string", "minLength": 1}}
                }
            }
        ]
    }

    try: validate(instance=merged, schema=SCHEMA_GENERATE_BY_PROMTPS)
    except ValidationError as e: return jsonify({"error": e.message}), 400

    model_id = merged["model_id"].strip()
    prompts = merged["prompts"]
    use_prc = merged["use_prc"]
    use_waterlo = merged["use_waterlo"]
    alpha = float(merged["alpha"])

    if use_prc: key_id = str(merged["key_id"])
    else: key_id = ""

    def on_generate_progress(current: int, total: int) -> None:

        emit("generate_prc_progress", {"current": current, "total": total})

    try:

        images = applyPRC(BASE_DIR, key_id, model_id, prompts, watermark=use_prc, out_path=None,
            on_progress=on_generate_progress)

        if use_waterlo: images = applyWaterLo(images, WATERLO_DIR, alpha=alpha, batch_size=BATCH_SIZE, out_path=None)

    except FileNotFoundError as e: return jsonify({"error": str(e)}), 400
    except ValueError as e: return jsonify({"error": str(e)}), 400

    b64_list = [_png_to_b64(im) for im in images]

    return jsonify({"images": b64_list, "count": len(b64_list)})


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)