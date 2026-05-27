import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.model_loader import ModelLoader
from api.routes import predict, sources, physics, stations


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Loading aeras checkpoint...")
    ModelLoader.load()
    print(f"[API] Model ready on {ModelLoader.device}")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="aeras API",
    description="Physics-Informed Neural Network for Delhi NCR PM2.5 prediction",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router,  prefix="/predict",  tags=["Prediction"])
app.include_router(sources.router,  prefix="/sources",  tags=["Source Localization"])
app.include_router(physics.router,  prefix="/physics",  tags=["Physics Parameters"])
app.include_router(stations.router, prefix="/stations", tags=["Stations"])


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ModelLoader.is_loaded(),
        "device": ModelLoader.device,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
