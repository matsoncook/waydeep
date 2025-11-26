from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from src.model_gemma_3_4b_it import Model

model = Model()
model.load()

app = FastAPI()
def split_into_chunks(s, chunk_size=10):
    return [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            input_str = await websocket.receive_text()
            # Simulate streaming
            print(f"input_str ", input_str)
            output_str = model.input(input_str=input_str)
            #output_str = 'def fibonacci(n):\n    if n==1:\n        return 1\n    else:\n        return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(10))\n\n#print(fibonacci(11))'
            print(f"output_str ",output_str)
            # for word in split_into_chunks(output_str):
            try:
                await websocket.send_text(output_str + "\n")
                await asyncio.sleep(0.3)
            except WebSocketDisconnect:
                print("Client disconnected during streaming.")
                return
    except WebSocketDisconnect:
        print("Client disconnected.")
