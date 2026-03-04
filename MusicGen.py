from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
from audiocraft.models import MusicGen
import io
import wave
import asyncio

app = FastAPI()
sessions = {}

print("Loading model...")
model = MusicGen.get_pretrained("facebook/musicgen-small", device='cpu')
model.set_generation_params(duration=5, use_sampling=True)
print("Model loaded")

def wavBytes(wav):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(model.sample_rate)
        f.writeframes((wav.cpu().numpy() * 32767).astype("int16").tobytes())
    buffer.seek(0)
    return buffer.read()

class Session:
    def __init__(self):
        self.clients = set()
        self.prompts = asyncio.Queue()
        self.started = False
        self.buffer = torch.empty((1, 0))
    
    async def send(self):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text("next")
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def generation(self):
        print("Generation started")
        await self.send()
        prompt = await self.prompts.get()
        wavfile = await asyncio.create_task(asyncio.to_thread(lambda:
             model.generate([prompt],model.sample_rate)))
        while True:
            await self.send()
            if not self.prompts.empty():
               prompt = await self.prompts.get()
            nextFile = await asyncio.create_task(asyncio.to_thread(lambda:
               model.generate([prompt],model.sample_rate)))
            first = wavfile.squeeze(0) 
            second = nextFile.squeeze(0)
            time = min(int(4 * model.sample_rate),first.shape[-1], second.shape[-1])
            overlap = ( first[..., -time:] * torch.linspace(1, 0, time) + second[..., :time] * torch.linspace(0, 1, time)) 
            finalV = torch.cat([ first[...,:-time], overlap ], dim=-1)
            if self.buffer.shape[-1] < 180 * model.sample_rate:
                self.buffer = torch.cat([self.buffer, finalV], dim=-1)
            print(self.buffer.numel())
            wavfile = nextFile[..., -time:]

    async def play(self,ws):
        sec = 44100 
        print("playing")
        while True:
            chunk = self.buffer[..., :min(sec,self.buffer.numel())]
            self.buffer = self.buffer[..., min(sec,self.buffer.numel()):]
            while chunk.numel() == 0:
                await asyncio.sleep(0.1)
            try:
                await ws.send_bytes(wavBytes(chunk.unsqueeze(0)))
                print("1 sec")
            except (WebSocketDisconnect, RuntimeError):
                break
            await asyncio.sleep(0.8)

def newSession(id: str) -> Session: 
    if id not in sessions: 
        sessions[id] = Session() 
    return sessions[id]

@app.websocket("/ws")
async def websocket_music(ws: WebSocket):
    await ws.accept()
    id = ws.query_params.get("session", "none")
    session = newSession(id)
    session.clients.add(ws)
    print("Connected new client on id:" + id)
    async def listen():
        while True:
            if ws.client_state.name == "CONNECTED":
                try:
                    command = await ws.receive_text()
                    if command == "play":
                        print("Start playing")
                        asyncio.create_task(session.play(ws))                    
                    else:
                        await session.prompts.put(command)
                        print(f"New prompt: {command}")
                except (WebSocketDisconnect):
                    print("bad connection")
                    session.clients.discard(ws)
                    break
            else:
                break
    if not session.started:
        session.started = True
        asyncio.create_task(session.generation())
    await listen()
        