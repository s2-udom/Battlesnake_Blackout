from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def index():
    return {
        "apiversion": "1",
        "author": "yourname",
        "color": "#ff0000",
        "head": "default",
        "tail": "default"
    }

@app.post("/start")
def start(data: dict):
    return {}

@app.post("/move")
def move(data: dict):
    return {"move": "up"}

@app.post("/end")
def end(data: dict):
    return {}
