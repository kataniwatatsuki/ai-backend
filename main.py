# main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from typing import Dict, List

# model imports (あなたの既存コードを流用)
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番はVercelドメインに限定推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- model (あなたの既存) ----------
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

# ---------------- predict ----------------
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

# ---------------- rooms & queues ----------------
# rooms: { room_id: { "members": {user: {"troubled":bool}}, "queues": [q1,q2,..] } }
rooms: Dict[str, Dict] = {}

def get_members(room_id: str):
    return [
        {"user": u, "troubled": info["troubled"]}
        for u, info in rooms.get(room_id, {}).get("members", {}).items()
    ]

async def broadcast(room_id: str, message: dict):
    if room_id not in rooms:
        return
    for q in list(rooms[room_id]["queues"]):
        await q.put(message)

# ---------------- SSE endpoint ----------------
@app.get("/events/{room}/{user}")
async def events_endpoint(request: Request, room: str, user: str):
    """
    Client connects to: GET /events/{room}/{user}
    On connect: user is added to room.members (troubled=False) and members list is broadcast.
    On disconnect: user removed and members broadcast.
    """
    if room not in rooms:
        rooms[room] = {"members": {}, "queues": []}

    # register member if not exist
    if user not in rooms[room]["members"]:
        rooms[room]["members"][user] = {"troubled": False}

    # create queue for this connection
    q: asyncio.Queue = asyncio.Queue()
    rooms[room]["queues"].append(q)

    # broadcast join + members
    await broadcast(room, {"type": "join", "user": user})
    await broadcast(room, {"type": "members", "users": get_members(room)})

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield f"event: message\ndata: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # keepalive ping for Cloudflare
                    yield "event: ping\ndata: keepalive\n\n"

                if await request.is_disconnected():
                    break
        finally:
            # cleanup on disconnect
            try:
                rooms[room]["queues"].remove(q)
            except Exception:
                pass
            # remove member and broadcast leave/members
            if user in rooms[room]["members"]:
                del rooms[room]["members"][user]
                await broadcast(room, {"type": "leave", "user": user})
                await broadcast(room, {"type": "members", "users": get_members(room)})
            # if no members and no queues, optionally delete room
            if not rooms[room]["members"] and not rooms[room]["queues"]:
                del rooms[room]

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return EventSourceResponse(event_generator(), headers=headers)

# ---------------- client → server endpoints ----------------
@app.post("/trouble")
async def trouble(data: dict):
    room = data.get("room")
    user = data.get("user")
    if not room or not user or room not in rooms or user not in rooms[room]["members"]:
        raise HTTPException(status_code=400, detail="invalid room/user")
    rooms[room]["members"][user]["troubled"] = True
    await broadcast(room, {"type": "trouble", "user": user, "message": "困っています！"})
    await broadcast(room, {"type": "members", "users": get_members(room)})
    return {"ok": True}

@app.post("/resolved")
async def resolved(data: dict):
    room = data.get("room")
    user = data.get("user")
    if not room or not user or room not in rooms or user not in rooms[room]["members"]:
        raise HTTPException(status_code=400, detail="invalid room/user")
    rooms[room]["members"][user]["troubled"] = False
    await broadcast(room, {"type": "members", "users": get_members(room)})
    await broadcast(room, {"type": "resolved", "user": user})
    return {"ok": True}

@app.post("/leave")
async def leave(data: dict):
    room = data.get("room")
    user = data.get("user")
    if not room or not user:
        raise HTTPException(status_code=400)
    if room in rooms and user in rooms[room]["members"]:
        del rooms[room]["members"][user]
        await broadcast(room, {"type": "leave", "user": user})
        await broadcast(room, {"type": "members", "users": get_members(room)})
    return {"ok": True}
