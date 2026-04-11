from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.api.ws.devices import device_ws
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from app.core.scheduler import offline_monitor_loop
from app.core.scheduler import repair_loop

from app.api.routes.devices import router as device_router
from app.api.routes.files import router as file_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes import chunks

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    offline_task = asyncio.create_task(offline_monitor_loop())
    repair_task = asyncio.create_task(repair_loop())

    print("Startup: Background loops started.")

    yield

    offline_task.cancel()
    repair_task.cancel()

    try:
        await offline_task
    except asyncio.CancelledError:
        print("Offline monitor stopped.")

    try:
        await repair_task
    except asyncio.CancelledError:
        print("Repair loop stopped.")


app = FastAPI(lifespan=lifespan)

# CORS – allow everything for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(device_router)
app.include_router(chunks.router)
app.include_router(file_router)
app.include_router(dashboard_router)

# Serve the dashboard frontend as static files
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
DASHBOARD_DIR.mkdir(exist_ok=True)
app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")

@app.websocket("/ws/device")
async def ws_service(websocket : WebSocket):
    await device_ws(websocket)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR:", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})