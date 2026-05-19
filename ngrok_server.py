from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import io
import os
import time
import uuid
from werkzeug.utils import secure_filename

# Rule-based insights engine (replaces Ollama/LLaMA)
from faw_rules import get_rule_based_insights

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "uploads"
MODEL_DIR = "visaia_models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenetv3_faw_vs_notfaw.pth")
YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")

if not os.path.exists(MOBILENET_PATH) or not os.path.exists(YOLO_PATH):
    raise FileNotFoundError("Model files missing in visaia_models")

# -----------------------------
# LOAD MOBILE NET
# -----------------------------
print("🚀 Loading MobileNetV3...")

mobilenet_model = models.mobilenet_v3_small(pretrained=False)
mobilenet_model.classifier[3] = torch.nn.Linear(
    mobilenet_model.classifier[3].in_features, 2
)

mobilenet_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
mobilenet_model.to(DEVICE)
mobilenet_model.eval()

CLASS_NAMES = ["FAW", "NotFAW"]

mobilenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# LOAD YOLO
# -----------------------------
print("🚀 Loading YOLOv8...")

yolo_model = YOLO(YOLO_PATH)

FAW_CLASSES = ["egg", "larva", "pupa", "moth"]

LIFE_STAGE_RISK = {
    "egg":   "Moderate",
    "larva": "High",
    "pupa":  "Low–Moderate",
    "moth":  "Moderate–High",
}

# -----------------------------
# IMAGE ANNOTATION
# -----------------------------
def annotate_image(image, detections):

    draw = ImageDraw.Draw(image, "RGBA")
    img_w, img_h = image.size

    COLORS = {
        "egg":   (255, 215, 0),
        "larva": (255, 0, 0),
        "pupa":  (255, 140, 0),
        "moth":  (0, 120, 255),
    }

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        stage = det["class"]
        conf  = det.get("confidence", 0)

        x1 = int(det["x"] * img_w)
        y1 = int(det["y"] * img_h)
        x2 = int((det["x"] + det["width"])  * img_w)
        y2 = int((det["y"] + det["height"]) * img_h)

        color = COLORS.get(stage, (255, 255, 255))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{stage.upper()} {conf * 100:.1f}%"

        bbox   = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        label_y = y1 - text_h - 6
        if label_y < 0:
            label_y = y1 + 4

        draw.rectangle(
            [x1, label_y, x1 + text_w + 6, label_y + text_h + 4],
            fill=(*color, 180),
        )
        draw.text(
            (x1 + 3, label_y + 2),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    return image


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return "<h1>FAW Server Running</h1>"


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file      = request.files["image"]
    img_bytes = file.read()

    filename    = secure_filename(file.filename)
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
    save_path   = os.path.join(UPLOAD_FOLDER, unique_name)

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # ── MobileNet binary classifier ──────────────────────────
        tensor = mobilenet_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = mobilenet_model(tensor)
            _, pred = torch.max(outputs, 1)

        prediction = CLASS_NAMES[pred.item()]

        # ── FAW detected → YOLO stage detection ──────────────────
        if prediction == "FAW":

            results = yolo_model(image)

            detected = []
            boxes    = []
            w, h     = image.size

            for r in results:
                for box in r.boxes:
                    cls  = FAW_CLASSES[int(box.cls[0])]
                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    boxes.append({
                        "class":      cls,
                        "x":          x1 / w,
                        "y":          y1 / h,
                        "width":      (x2 - x1) / w,
                        "height":     (y2 - y1) / h,
                        "confidence": conf,
                    })
                    detected.append(cls)

            # Annotate and save image
            annotated  = annotate_image(image.copy(), boxes)
            out_name   = "annotated_" + unique_name
            annotated.save(os.path.join(UPLOAD_FOLDER, out_name))

            # Overall risk: High if any high-risk stage is present
            risk = (
                "High"
                if any(LIFE_STAGE_RISK.get(d, "Low") in ("High", "Moderate–High")
                       for d in detected)
                else "Low–Moderate"
            )

            # ── Rule-based insights (replaces Ollama) ────────────
            insights = get_rule_based_insights(detected, risk)

            return jsonify({
                "pest":       "Fall Army Worm",
                "stages":     detected,
                "risk":       risk,
                "boxes":      boxes,
                "analysis":   insights["analysis"],
                "treatment":  insights["treatment"],
                "image_url":  f"http://{request.host}/uploads/{out_name}",
            })

        # ── NOT FAW ──────────────────────────────────────────────
        else:
            return jsonify({
                "pest":      "Unknown / Not FAW",
                "analysis":  (
                    "The image does not appear to contain Fall Armyworm (FAW). "
                    "The MobileNetV3 classifier did not detect FAW characteristics. "
                    "If you believe this is incorrect, try a clearer, closer image "
                    "of the pest or affected plant part."
                ),
                "treatment": (
                    "No FAW-specific treatment is required at this time. Continue "
                    "regular field scouting using the BPI W-pattern method "
                    "(5 inspection points per field, minimum 10 plants per stop). "
                    "Maintain pheromone traps at 4–5 traps per hectare and replace "
                    "lures every 6 weeks. If FAW is suspected based on crop damage "
                    "symptoms (window-pane feeding, whorl damage, inverted-Y markings "
                    "on larvae), consult your Regional Crop Protection Center (RCPC)."
                ),
                "image_url": f"http://{request.host}/uploads/{unique_name}",
            })

    except Exception as e:
        print("❌ SERVER ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------
# RUN — NGROK COMPATIBLE (PORT 8000)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)