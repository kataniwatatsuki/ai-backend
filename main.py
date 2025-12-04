from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio

import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
from typing import Dict
from uuid import uuid4
import json

# -------------------------
# FastAPI 本体
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# モデルロード
# -------------------------
MODEL_PATH = "models/fine_tuned_from_efficientnet_b0_best.pth"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

num_classes = len(CLASS_NAMES)
model = EfficientNet.from_name("efficientnet-b0")
in_features = model._fc.in_features
model._fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

mp_face_detection = mp.solutions.face_detection


def detect_face(img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as fd:
        return fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def predict_expression(face_img):
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred = torch.max(probs, 0)
    return CLASS_NAMES[pred.item()] if confidence >= 0.6 else "neutral"


# -------------------------
# /predict 表情認識API
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    results = detect_face(img)

    if results and results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = img.shape
        x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
        bw, bh = int(bboxC.width * w), int(bboxC.height * h)

        if bw < 80 or bh < 80:
            return {"expression": "neutral", "face": None}

        face_img = img[max(0, y):min(y + bh, h), max(0, x):min(x + bw, w)]
        expression = predict_expression(face_img)
        return {"expression": expression, "face": {"x": x, "y": y, "width": bw, "height": bh}}

    return {"expression": "neutral", "face": None}


# -------------------------------------
# SSE (Server-Sent Events) ルーム管理
# -------------------------------------

# ルームデータ
rooms: Dict[str, Dict] = {}   # {room_id: {"members": {}, "queue": asyncio.Queue()} }


def get_members(room_id: str):
    return [
        {"user": m["user"], "troubled": m["troubled"]}
        for m in rooms.get(room_id, {}).get("members", {}).values()
    ]


async def broadcast(room_id: str, message: dict):
    """指定ルームへイベントを配信"""
    if room_id in rooms:
        await rooms[room_id]["queue"].put(message)


# -------------------------
# SSE 接続
# -------------------------
@app.get("/events")
async def events(request: Request, room: str):
    if room not in rooms:
        rooms[room] = {"members": {}, "queue": asyncio.Queue()}

    queue = rooms[room]["queue"]

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            message = await queue.get()
            yield {
                "event": "message",
                "data": json.dumps(message)   # ← 修正
            }

    return EventSourceResponse(event_generator())



# -------------------------
# 入室
# -------------------------
@app.post("/join")
async def join(data: dict):
    room = data["room"]
    user = data["user"]
    sid = str(uuid4())

    if room not in rooms:
        rooms[room] = {"members": {}, "queue": asyncio.Queue()}

    rooms[room]["members"][sid] = {"user": user, "troubled": False}

    await broadcast(room, {"type": "join", "user": user})
    await broadcast(room, {"type": "members", "users": get_members(room)})

    return {"sid": sid}


# -------------------------
# 困った
# -------------------------
@app.post("/trouble")
async def trouble(data: dict):
    room = data["room"]
    sid = data["sid"]

    rooms[room]["members"][sid]["troubled"] = True

    await broadcast(room, {"type": "trouble", "user": rooms[room]["members"][sid]["user"]})
    await broadcast(room, {"type": "members", "users": get_members(room)})

    return {"ok": True}


# -------------------------
# 解決
# -------------------------
@app.post("/resolved")
async def resolved(data: dict):
    room = data["room"]
    sid = data["sid"]

    rooms[room]["members"][sid]["troubled"] = False

    await broadcast(room, {"type": "members", "users": get_members(room)})

    return {"ok": True}
