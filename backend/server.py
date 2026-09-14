import asyncio
import json
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

allow_origins = ["https://kivilcimlab.org"]
MAX_CONNECTIONS = int(os.environ.get("MAX_CONNECTIONS", "8"))
MAX_NODES = int(os.environ.get("MAX_NODES", "3500"))
MAX_MESSAGE_BYTES = int(os.environ.get("MAX_MESSAGE_BYTES", "262144"))
active_connections = 0

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def origin_allowed(origin: str | None) -> bool:
    if "*" in allow_origins:
        return True
    if not origin:
        return False
    return origin.rstrip("/") in allow_origins


def finite_pair(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True}


class Positions(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.coords = nn.Embedding(num_nodes, 2)
        self.coords.weight.data.normal_(0.5, 0.05)

    def forward(self, indices):
        return self.coords(indices)


async def train_loop(websocket: WebSocket, model: Positions, optimiser: optim.Optimizer, target_tensor: torch.Tensor):
    num_nodes = target_tensor.shape[0]
    indices = torch.arange(num_nodes)

    try:
        while True:
            optimiser.zero_grad()
            preds = model(indices)
            loss = nn.MSELoss(reduction="sum")(preds, target_tensor)
            loss.backward()
            optimiser.step()

            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()

            await websocket.send_text(json.dumps({
                "loss": loss.item() / num_nodes,
                "grad_norm": grad_norm,
                "preds": preds.detach().cpu().tolist(),
            }))
            await asyncio.sleep(0.066)
    except Exception:
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections

    if not origin_allowed(websocket.headers.get("origin")):
        await websocket.close(code=1008)
        return

    if active_connections >= MAX_CONNECTIONS:
        await websocket.close(code=1013)
        return

    await websocket.accept()
    active_connections += 1

    training_task = None
    model = None

    try:
        while True:
            raw = await websocket.receive_text()
            if len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
                continue

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(payload, dict):
                continue

            kind = payload.get("type")

            if kind == "init":
                targets = payload.get("targets")
                if not isinstance(targets, list) or not targets or len(targets) > MAX_NODES:
                    continue

                cleaned = []
                for item in targets:
                    pair = finite_pair(item)
                    if pair is None:
                        cleaned = []
                        break
                    cleaned.append(pair)
                if not cleaned:
                    continue

                if training_task:
                    training_task.cancel()

                target_tensor = torch.tensor(cleaned, dtype=torch.float32)
                model = Positions(target_tensor.shape[0])
                optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
                training_task = asyncio.create_task(
                    train_loop(websocket, model, optimiser, target_tensor)
                )

            elif kind == "disrupt" and model is not None:
                data = payload.get("data")
                if not isinstance(data, list):
                    continue

                n = model.coords.num_embeddings
                with torch.no_grad():
                    for item in data[:n]:
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        try:
                            idx = int(item[0])
                        except (TypeError, ValueError):
                            continue
                        if idx < 0 or idx >= n:
                            continue
                        pair = finite_pair(item[1:3])
                        if pair is None:
                            continue
                        model.coords.weight.data[idx] = torch.tensor(pair, dtype=torch.float32)

    except WebSocketDisconnect:
        pass
    finally:
        active_connections = max(0, active_connections - 1)
        if training_task:
            training_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
