from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from typing import Dict, List
import os

app = FastAPI()

API_KEY = os.getenv("kannkyou-api")  # ← 環境変数を取得
print(f"API_KEY: {API_KEY}")
# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/fine_tuned_from_efficientnet_b0_best.pth"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# EfficientNet モデル読み込み
num_classes = len(CLASS_NAMES)
model = EfficientNet.from_name("efficientnet-b0")
in_features = model._fc.in_features
model._fc = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 前処理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== OpenCV DNN 顔検出（res10_300x300_ssd_iter_140000.caffemodel） =====
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

def detect_face(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1 - x, y1 - y))
    return faces

def predict_expression(face_img):
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, pred = torch.max(probs, 0)
    if confidence < 0.6:
        return "neutral"
    return CLASS_NAMES[pred.item()]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    faces = detect_face(img)

    if faces:
        x, y, bw, bh = faces[0]
        # numpy.int64 → int に変換
        x, y, bw, bh = int(x), int(y), int(bw), int(bh)

        if bw < 80 or bh < 80:
            return {"expression": "neutral", "face": None}

        face_img = img[max(0, y):min(y+bh, img.shape[0]), max(0, x):min(x+bw, img.shape[1])]
        expression = predict_expression(face_img)

        return {
            "expression": expression,
            "face": {"x": x, "y": y, "width": bw, "height": bh}
        }

    return {"expression": "平常", "face": None}

# ======== WebSocket 部分（変更なし） ========
rooms: Dict[str, List[Dict]] = {}

@app.websocket("/ws/{room_id}/{username}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = []

    user_data = {"ws": websocket, "user": username, "troubled": False}
    rooms[room_id].append(user_data)

    async def broadcast_members():
        users_info = [{"user": c["user"], "troubled": c.get("troubled", False)} for c in rooms[room_id]]
        for client in rooms[room_id]:
            await client["ws"].send_json({"type": "members", "users": users_info})

    for client in rooms[room_id]:
        await client["ws"].send_json({"type": "join", "user": username})

    await broadcast_members()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "trouble":
                for c in rooms[room_id]:
                    if c["user"] == username:
                        if c["troubled"]:
                            break
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
