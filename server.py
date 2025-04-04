from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Simulate streaming
            for word in "Hello from the bot!".split():
                try:
                    await websocket.send_text(word + " ")
                    await asyncio.sleep(0.3)
                except WebSocketDisconnect:
                    print("Client disconnected during streaming.")
                    return
    except WebSocketDisconnect:
        print("Client disconnected.")
