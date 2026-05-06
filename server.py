from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms, models
from ultralytics import YOLO
import io
import requests
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Model Paths
# -----------------------------
MODEL_DIR = "visaia_models"
MOBILENET_PATH = os.path.join(MODEL_DIR, "mobilenetv3_faw_vs_notfaw.pth")
YOLO_PATH = os.path.join(MODEL_DIR, "best.pt")

if not os.path.exists(MOBILENET_PATH) or not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"Check if models exist in {MODEL_DIR}")

# -----------------------------
# Load MobileNet
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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Load YOLO
# -----------------------------
print("🚀 Loading YOLOv8...")

yolo_model = YOLO(YOLO_PATH)

FAW_CLASSES = ["egg","larva","pupa","moth"]

LIFE_STAGE_RISK = {
    "egg":"Low",
    "larva":"High",
    "pupa":"Low",
    "moth":"High"
}

# -----------------------------
# Annotation Function
# -----------------------------
def annotate_image(image, detections):

    draw = ImageDraw.Draw(image,"RGBA")

    img_w, img_h = image.size

    COLOR_MAP = {
        "egg": (255,215,0),
        "larva": (255,0,0),
        "pupa": (255,140,0),
        "moth": (0,120,255)
    }

    try:
        font = ImageFont.truetype("arial.ttf",18)
    except:
        font = ImageFont.load_default()

    for det in detections:

        stage = det["class"]
        confidence = det.get("confidence",0)

        x = det["x"] * img_w
        y = det["y"] * img_h
        w = det["width"] * img_w
        h = det["height"] * img_h

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        color = COLOR_MAP.get(stage,(255,255,255))

        fill_color = (*color,70)

        draw.rectangle([x1,y1,x2,y2],fill=fill_color)
        draw.rectangle([x1,y1,x2,y2],outline=color,width=3)

        label = f"{stage.capitalize()} {int(confidence*100)}%"

        bbox = draw.textbbox((0,0),label,font=font)
        text_w = bbox[2]-bbox[0]
        text_h = bbox[3]-bbox[1]

        label_x1 = x1
        label_y1 = y1 - text_h - 6

        if label_y1 < 0:
            label_y1 = y1 + 4

        label_x2 = label_x1 + text_w + 6
        label_y2 = label_y1 + text_h + 4

        draw.rectangle(
            [label_x1,label_y1,label_x2,label_y2],
            fill=(0,0,0,160)
        )

        draw.text(
            (label_x1+3,label_y1+2),
            label,
            fill=(255,255,255),
            font=font
        )

    return image

# -----------------------------
# OpenRouter LLM
# -----------------------------
def ask_llm(image_bytes):

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        return "OpenRouter API key missing"

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    url = "https://openrouter.ai/api/v1/chat/completions"

    models_to_try = [
        "google/gemma-3-27b-it:free",
        "qwen/qwen-2-vl-7b-instruct:free",
        "nvidia/llama-3.2-nv-vision-70b:free"
    ]

    for model in models_to_try:

        try:

            payload = {
                "model":model,
                "messages":[
                    {
                        "role":"user",
                        "content":[
                            {"type":"text","text":"Identify this pest: common and scientific name."},
                            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            }

            response = requests.post(
                url,
                headers={
                    "Authorization":f"Bearer {api_key}",
                    "HTTP-Referer":"http://localhost:5000",
                    "X-Title":"FAW Detection"
                },
                json=payload,
                timeout=12
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]

        except:
            continue

    return "AI analysis unavailable."

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return "<h1>FAW Server Online</h1>"

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],filename)

@app.route("/predict",methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"}),400

    file = request.files["image"]
    img_bytes = file.read()

    filename = secure_filename(file.filename)

    import time,uuid

    unique_filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"],unique_filename)

    with open(save_path,"wb") as f:
        f.write(img_bytes)

    try:

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # -----------------------------
        # MobileNet classification
        # -----------------------------
        input_tensor = mobilenet_transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = mobilenet_model(input_tensor)
            _,pred = torch.max(outputs,1)

        class_prediction = CLASS_NAMES[pred.item()]

        # -----------------------------
        # If FAW → Run YOLO
        # -----------------------------
        if class_prediction == "FAW":

            results = yolo_model(image)

            detected_classes = []
            boxes_data = []

            width,height = image.size

            for result in results:

                for box in result.boxes:

                    cls_idx = int(box.cls[0])
                    stage = FAW_CLASSES[cls_idx]

                    conf = float(box.conf[0])

                    detected_classes.append(stage)

                    x1,y1,x2,y2 = box.xyxy[0].tolist()

                    boxes_data.append({
                        "class":stage,
                        "x":x1/width,
                        "y":y1/height,
                        "width":(x2-x1)/width,
                        "height":(y2-y1)/height,
                        "confidence":conf
                    })

            # Annotate image
            annotated = annotate_image(image.copy(),boxes_data)

            annotated_name = "annotated_"+unique_filename
            annotated_path = os.path.join(app.config["UPLOAD_FOLDER"],annotated_name)

            annotated.save(annotated_path)

            stage_text = ", ".join(detected_classes) if detected_classes else "None"
            
            risk_level = "Low"
            for stage in detected_classes:
                if LIFE_STAGE_RISK.get(stage) == "High":
                    risk_level = "High"
                    break

            return jsonify({
                "pest":"Fall Army Worm",
                "stage":stage_text,
                "risk": risk_level,
                "boxes":boxes_data,
                "image_url":f"http://{request.host}/uploads/{annotated_name}"
            })

        # -----------------------------
        # If Not FAW → LLM
        # -----------------------------
        else:

            ai_result = ask_llm(img_bytes)

            return jsonify({
                "pest":"Unknown",
                "prediction":ai_result,
                "image_url":f"http://{request.host}/uploads/{unique_filename}"
            })

    except Exception as e:
        return jsonify({"error":str(e)}),500


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True,threaded=True)