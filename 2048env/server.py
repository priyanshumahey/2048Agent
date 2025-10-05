import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from main import Game2048Env

app = FastAPI(title="2048 Game Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

connections = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "active_sessions": len(connections),
        "message": "2048 Game Server is running"
    }

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "actie_sessions": list(connections.keys()),
        "count": len(connections)
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    """
    Websocket endpoint for game interaction

    Message format:
    - Client: {"action": [0,3]}
    - Client: {"command": "reset"}
    - Server: game state updates with type, state, score, etc
    """

    await ws.accept()
    print(f"Session {session_id} connected")

    env = Game2048Env()
    obs, _ = env.reset()
    
    connections[session_id] = {"socket": ws, "env": env}

    await ws.send_json({
        "type": "init",
        "state": obs.tolist(),
        "score": env.score,
        "max_tile": int(obs.max()),
        "done": False,
        "message": "Game started!"
    })

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("command") == "reset":
                obs, _ = env.reset()
                await ws.send_json({
                    "type": "reset",
                    "state": obs.tolist(),
                    "score": env.score,
                    "max_tile": int(obs.max()),
                    "done": False,
                    "message": "Game reset"
                })
                continue

            action = data.get("action", None)
            if action is None:
                await ws.send_json({
                    "type": "error",
                    "message": "No action provided. Send {\"action\": 0-3}"
                })
                continue
            
            if not isinstance(action, int) or action not in [0, 1, 2, 3]:
                await ws.send_json({
                    "type": "error",
                    "message": f"Invalid action: {action}. Must be 0 (up), 1 (down), 2 (left), or 3 (right)"
                })
                continue
                
            obs, reward, done, _, info = env.step(action)

            update = {
                "type": "step",
                "action": action,
                "action_name": ["UP", "DOWN", "LEFT", "RIGHT"][action],
                "state": obs.tolist(),
                "reward": int(reward),
                "score": env.score,
                "max_tile": int(obs.max()),
                "done": done,
                "valid_move": info["valid_move"]
            }
            await ws.send_json(update)

            if done:
                print(f"Session {session_id}: Game Over! Score: {env.score}, Max tile: {obs.max()}")
                await asyncio.sleep(1)
                
                obs, _ = env.reset()
                await ws.send_json({
                    "type": "reset",
                    "state": obs.tolist(),
                    "score": env.score,
                    "max_tile": int(obs.max()),
                    "done": False,
                    "message": "Auto-reset after game over"
                })

    except WebSocketDisconnect:
        print(f"✗ Session {session_id} disconnected")
        if session_id in connections:
            del connections[session_id]
    except Exception as e:
        print(f"Error in session {session_id}: {e}")
        if session_id in connections:
            del connections[session_id]


if __name__ == "__main__":
    print("Server started")
    print("Server running at http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/{session_id}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
