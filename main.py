from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
from typing import Dict
import socketio

# ===== Socket.IO サーバー =====
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)
app = FastAPI()
app.mount("/socket.io", socketio.ASGIApp(sio))

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== EfficientNet モデル =====
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
    if confidence < 0.6:
        return "neutral"
    return CLASS_NAMES[pred.item()]

# ========================
# /predict 表情認識API
# ========================
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


# ========================
# Socket.IO ルーム管理
# ========================
rooms: Dict[str, Dict[str, Dict]] = {}

def get_members(room_id: str):
    return [
        {"user": data["user"], "troubled": data["troubled"]}
        for data in rooms.get(room_id, {}).values()
    ]

# ===== 接続 =====
@sio.event
async def connect(sid, environ):
    print(f"🔌 Connected: {sid}")
    await sio.save_session(sid, {})

# ===== 切断 =====
@sio.event
async def disconnect(sid):
    print(f"❌ Disconnected: {sid}")
    for room_id, users in list(rooms.items()):
        if sid in users:
            username = users[sid]["user"]
            del users[sid]
            if not users:
                del rooms[room_id]
            await sio.emit("leave", {"user": username}, room=room_id)
            await sio.emit("members", {"users": get_members(room_id)}, room=room_id)
            break

# ===== 入室 =====
@sio.event
async def join_room(sid, data):
    room_id = data["room"]
    username = data["user"]

    await sio.enter_room(sid, room_id)

    if room_id not in rooms:
        rooms[room_id] = {}

    rooms[room_id][sid] = {"user": username, "troubled": False}

    await sio.emit("join", {"user": username}, room=room_id)
    await sio.emit("members", {"users": get_members(room_id)}, room=room_id)

# ===== 困った通知 =====
@sio.event
async def trouble(sid, data):
    room_id = data["room"]
    if room_id in rooms and sid in rooms[room_id]:
        rooms[room_id][sid]["troubled"] = True
        await sio.emit("members", {"users": get_members(room_id)}, room=room_id)
        await sio.emit("trouble", {"user": rooms[room_id][sid]["user"]}, room=room_id)

# ===== 解決通知 =====
@sio.event
async def resolved(sid, data):
    room_id = data["room"]
    if room_id in rooms and sid in rooms[room_id]:
        rooms[room_id][sid]["troubled"] = False
        await sio.emit("members", {"users": get_members(room_id)}, room=room_id)
