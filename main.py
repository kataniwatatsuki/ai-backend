from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from typing import Dict, List
import asyncio
import os

app = FastAPI()

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# 絶対パス設定（Render で必須）
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

FACE_PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
EXPR_MODEL_PATH = os.path.join(MODEL_DIR, "fine_tuned_from_efficientnet_b0_best.pth")

# ==============================
# 表情分類モデル
# ==============================
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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==============================
# OpenCV DNN 顔検出モデル読み込み
# ==============================
#!!! ここが重要 → 正しい順番
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)


def detect_face(img):
    """OpenCV DNNを使った顔検出（最大信頼度の顔のみ）"""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    max_conf = 0
    best_box = None

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.5 and conf > max_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype(int)
            max_conf = conf

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "conf": max_conf}


def predict_expression(face_img):
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred = torch.max(probs, 0)
    if confidence < 0.6:
        return "neutral"
    return CLASS_NAMES[pred.item()]


# ==============================
# /predict
# ==============================
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

    face_img = img[max(0, y): min(y + h, H), max(0, x): min(x + w, W)]

    if face_img.size == 0:
        return {"expression": "neutral", "face": None}

    expression = predict_expression(face_img)
    return {
        "expression": expression,
        "face": {"x": x, "y": y, "width": w, "height": h}
    }


# ==============================
# WebSocket（元のまま）
# ==============================
rooms: Dict[str, List[Dict]] = {}


@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = []

    user_data = {"ws": websocket, "user": username, "troubled": False}
    rooms[room_id].append(user_data)

    async def broadcast_members():
        users_info = [{"user": c["user"], "troubled": c["troubled"]} for c in rooms[room_id]]
        for client in rooms[room_id]:
            await client["ws"].send_json({"type": "members", "users": users_info})

    # 入室通知
    for client in rooms[room_id]:
        await client["ws"].send_json({"type": "join", "user": username})

    await broadcast_members()

    async def ping_loop():
        while True:
            try:
                await websocket.send_json({"type": "ping"})
            except:
                break
            await asyncio.sleep(15)

    ping_task = asyncio.create_task(ping_loop())

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "trouble":
                for c in rooms[room_id]:
                    if c["user"] == username:
                        c["troubled"] = True
                await broadcast_members()

                for client in rooms[room_id]:
                    await client["ws"].send_json({
                        "type": "trouble",
                        "user": username,
                        "message": "困っています！"
                    })

            if data["type"] == "resolved":
                for c in rooms[room_id]:
                    if c["user"] == username:
                        c["troubled"] = False
                await broadcast_members()

    except WebSocketDisconnect:
        rooms[room_id] = [c for c in rooms[room_id] if c["ws"] != websocket]
        for client in rooms[room_id]:
            await client["ws"].send_json({"type": "leave", "user": username})
        await broadcast_members()
    finally:
        ping_task.cancel()
