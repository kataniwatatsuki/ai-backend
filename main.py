# main.py
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
import json
from uuid import uuid4
from typing import Dict

# --- image/model imports (your existing model parts) ---
import cv2
import numpy as np
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import mediapipe as mp

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番では制限することを検討
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Model (same as you had)
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
# /predict endpoint (unchanged)
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

# -------------------------
# Room management for SSE
# -------------------------
# rooms: { room_id: { "members": {sid: {user, troubled}}, "queues": [asyncio.Queue(), ...] } }
rooms: Dict[str, Dict] = {}

def get_members(room_id: str):
    return [
        {"user": m["user"], "troubled": m["troubled"]}
        for m in rooms.get(room_id, {}).get("members", {}).values()
    ]

async def broadcast_to_room(room_id: str, message: dict):
    """Put message into every queue for that room."""
    if room_id not in rooms:
        return
    for q in rooms[room_id]["queues"]:
        await q.put(message)

# -------------------------
# Events endpoint (SSE)
# -------------------------
@app.get("/events")
async def events(request: Request, room: str, sid: str = None):
    """
    Client connects to: /events?room=ROOM&sid=SID
    - SID is optional; provided after /join returns it.
    """
    if room not in rooms:
        # initialize
        rooms[room] = {"members": {}, "queues": []}

    queue: asyncio.Queue = asyncio.Queue()
    rooms[room]["queues"].append(queue)

    async def send_loop():
        try:
            while True:
                # wait for next message, but timeout for keepalive ping
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=10.0)
                    # push as plain message (client expects JSON in data)
                    yield f"event: message\ndata: {json.dumps(msg, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # send keepalive ping to prevent Cloudflare closing idle stream
                    yield "event: ping\ndata: keepalive\n\n"

                # detect client disconnect
                if await request.is_disconnected():
                    break
        except asyncio.CancelledError:
            pass
        finally:
            # remove queue on disconnect
            try:
                rooms[room]["queues"].remove(queue)
            except Exception:
                pass

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable buffering for nginx-like proxies
    }

    return EventSourceResponse(send_loop(), headers=headers)

# -------------------------
# join endpoint - returns sid
# -------------------------
@app.post("/join")
async def join(data: dict):
    """
    body: { "room": "...", "user": "..." }
    returns: { "sid": "..." }
    """
    room = data.get("room")
    user = data.get("user")
    if not room or not user:
        raise HTTPException(status_code=400, detail="room and user required")

    sid = str(uuid4())
    if room not in rooms:
        rooms[room] = {"members": {}, "queues": []}

    rooms[room]["members"][sid] = {"user": user, "troubled": False}

    # Broadcast join & members list
    await broadcast_to_room(room, {"type": "join", "user": user})
    await broadcast_to_room(room, {"type": "members", "users": get_members(room)})

    return {"sid": sid}

# -------------------------
# leave endpoint (optional)
# -------------------------
@app.post("/leave")
async def leave(data: dict):
    room = data.get("room")
    sid = data.get("sid")
    if not room or not sid:
        raise HTTPException(status_code=400)
    if room in rooms and sid in rooms[room]["members"]:
        username = rooms[room]["members"][sid]["user"]
        del rooms[room]["members"][sid]
        # broadcast leave + members
        await broadcast_to_room(room, {"type": "leave", "user": username})
        await broadcast_to_room(room, {"type": "members", "users": get_members(room)})
    return {"ok": True}

# -------------------------
# trouble / resolved
# -------------------------
@app.post("/trouble")
async def trouble(data: dict):
    room = data.get("room")
    sid = data.get("sid")
    if not room or not sid or room not in rooms or sid not in rooms[room]["members"]:
        raise HTTPException(status_code=400)
    rooms[room]["members"][sid]["troubled"] = True
    await broadcast_to_room(room, {"type": "trouble", "user": rooms[room]["members"][sid]["user"]})
    await broadcast_to_room(room, {"type": "members", "users": get_members(room)})
    return {"ok": True}

@app.post("/resolved")
async def resolved(data: dict):
    room = data.get("room")
    sid = data.get("sid")
    if not room or not sid or room not in rooms or sid not in rooms[room]["members"]:
        raise HTTPException(status_code=400)
    rooms[room]["members"][sid]["troubled"] = False
    await broadcast_to_room(room, {"type": "members", "users": get_members(room)})
    return {"ok": True}
