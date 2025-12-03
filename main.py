from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import asyncio
import json
import os
from typing import Dict, List

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/健康")
def health_check():
    return {"status": "ok"}
# ===== ここまで画像認識は変更なし =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EXPR_MODEL_PATH = os.path.join(MODEL_DIR, "fine_tuned_from_efficientnet_b0_best.pth")

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_classes = len(CLASS_NAMES)

model = EfficientNet.from_name("efficientnet-b0")
in_features = model._fc.in_features
model._fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(EXPR_MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    max_conf = 0
    best_box = None

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf > 0.5 and conf > max_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)
            max_conf = conf

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return {"x": int(x1), "y": int(y1), "w": int(x2 - x1), "h": int(y2 - y1)}


def predict_expression(face_img):
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred = torch.max(probs, 0)
    if float(confidence) < 0.6:
        return "neutral"
    return CLASS_NAMES[int(pred.item())]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    box = detect_face(img)
    if not box:
        return {"expression": "neutral", "face": None}

    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    H, W, _ = img.shape
    face_img = img[max(0, y):min(y+h, H), max(0, x):min(x+w, W)]

    if face_img.size == 0:
        return {"expression": "neutral", "face": None}

    expression = predict_expression(face_img)

    face = {"x": x, "y": y, "width": w, "height": h}
    return {"expression": expression, "face": face}


# ==============================
# 🔵 SSE 用データ管理
# ==============================
rooms: Dict[str, List[Dict]] = {}
listeners: Dict[str, List[asyncio.Queue]] = {}  # ← SSE クライアントごとの Queue


def broadcast(room_id: str, message: dict):
    """特定の部屋に SSE でブロードキャスト"""
    if room_id not in listeners:
        return
    for q in listeners[room_id]:
        q.put_nowait(message)


# ==============================
# 🔵 クライアント → サーバー
# ==============================
@app.post("/trouble/{room_id}/{username}")
async def trouble(room_id: str, username: str):
    """困っている発火"""
    for u in rooms.get(room_id, []):
        if u["user"] == username:
            u["troubled"] = True

    broadcast(room_id, {"type": "members", "users": rooms[room_id]})
    broadcast(room_id, {"type": "trouble", "user": username})
    return {"status": "ok"}


@app.post("/resolve/{room_id}/{username}")
async def resolve(room_id: str, username: str):
    """困っている解除"""
    for u in rooms.get(room_id, []):
        if u["user"] == username:
            u["troubled"] = False

    broadcast(room_id, {"type": "members", "users": rooms[room_id]})
    return {"status": "ok"}


@app.post("/join/{room_id}/{username}")
async def join(room_id: str, username: str):
    """部屋に参加"""
    rooms.setdefault(room_id, [])
    listeners.setdefault(room_id, [])

    if not any(u["user"] == username for u in rooms[room_id]):
        rooms[room_id].append({"user": username, "troubled": False})

    broadcast(room_id, {"type": "members", "users": rooms[room_id]})
    broadcast(room_id, {"type": "join", "user": username})
    return {"status": "joined"}


# ==============================
# 🔵 サーバー → クライアント（SSE）
# ==============================
@app.get("/stream/{room_id}")
async def stream(room_id: str):
    """SSE ストリーム"""
    q = asyncio.Queue()
    listeners.setdefault(room_id, [])
    listeners[room_id].append(q)

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                yield f"data: {json.dumps(msg)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            listeners[room_id].remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
