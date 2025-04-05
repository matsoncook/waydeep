from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from src.model_very_small import Model

model = Model()
model.load()

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            input_str = await websocket.receive_text()
            # Simulate streaming
            print(f"input_str ", input_str)
            output_str = model.input(input_str=input_str)
            print(f"output_str ",output_str)
            for word in output_str.split("\n"):
                try:
                    await websocket.send_text(word + "\n")
                    await asyncio.sleep(0.3)
                except WebSocketDisconnect:
                    print("Client disconnected during streaming.")
                    return
    except WebSocketDisconnect:
        print("Client disconnected.")
