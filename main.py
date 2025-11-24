from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp
from typing import Dict, List

app = FastAPI()

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

mp_face_detection = mp.solutions.face_detection

def detect_face(img):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as fd:
        return fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def predict_expression(face_img):
    img = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

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
        face_img = img[max(0, y):min(y + bh, h), max(0, x):min(x + bw, w)]
        expression = predict_expression(face_img)
        return {"expression": expression, "face": {"x": x, "y": y, "width": bw, "height": bh}}
    return {"expression": "平常", "face": None}

# ======== WebSocket ========
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

    # 入室通知
    for client in rooms[room_id]:
        await client["ws"].send_json({"type": "join", "user": username})

    await broadcast_members()

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "trouble":
                for c in rooms[room_id]:
                    if c["user"] == username:
                        c["troubled"] = True
                await broadcast_members()
                # 個別通知
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
