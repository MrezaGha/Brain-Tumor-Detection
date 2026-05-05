"""
main.py — FastAPI entry point for the Brain Tumor AI Agent
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="Brain Tumor AI Agent",
    description="Agentic AI assistant for symptom intake and MRI classification",
    version="1.0.0",
)

# Allow local frontend dev if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (CSS/JS if ever extracted to separate files)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Include all API routes under /api prefix
app.include_router(router, prefix="/api")


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
