"""
FastAPI layer. Run with:
    uvicorn src.api.main:app --reload

This wraps the same run_brochure_workflow()/run_blog_workflow() functions
the Streamlit app and the CLI scripts use — it exists so the platform can
eventually be called from other services. For local development, calling
the workflow directly (as streamlit_app.py does) is simpler and is not
required to go through here.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.brochure import catalog
from src.core import win_asyncio
from src.core.config import settings
from src.core.ids import brief_dir
from src.data import blog_repository, repository
from src.data.ingestion import load_property_input
from src.data.schemas import BlogGenerationRecord, GenerationRecord
from src.workflows.blog import run_blog_workflow
from src.workflows.brochure import run_brochure_workflow

# Windows-only: keep closed-connection tracebacks out of the log.
win_asyncio.install()

app = FastAPI(title="Agentic Content Platform", version="0.1.0")


class GenerateBrochureRequest(BaseModel):
    property_id: str
    # One of the nine templates by id ("mid_portrait") or name ("Mid-Tier -
    # Portrait"). Left empty, the property's own category is used with
    # `orientation`.
    template_name: Optional[str] = None
    orientation: str = "landscape"     # landscape | portrait | digital


class GenerateBlogRequest(BaseModel):
    blog_id: str


class HumanReviewRequest(BaseModel):
    status: str  # "approved" | "revision_requested"
    comment: str = ""


# ---------------------------------------------------------------------------
# Brochure Generation
# ---------------------------------------------------------------------------

@app.post("/generate/brochure", response_model=GenerationRecord)
def generate_brochure(req: GenerateBrochureRequest) -> GenerationRecord:
    try:
        property_dir = brief_dir(settings.data_dir_resolved / "properties", req.property_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not property_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown property_id '{req.property_id}'")
    prop = load_property_input(property_dir)
    spec = catalog.resolve(req.template_name, prop.category, req.orientation)
    return run_brochure_workflow(property_dir, template_name=spec.id)


@app.get("/generations/{generation_id}", response_model=GenerationRecord)
def get_generation(generation_id: str) -> GenerationRecord:
    record = repository.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    return record


@app.post("/generations/{generation_id}/review", response_model=GenerationRecord)
def review_generation(generation_id: str, req: HumanReviewRequest) -> GenerationRecord:
    """Layer 3 (Human Review): records a person's approve/request-revision
    decision against an existing generation."""
    if req.status not in {"approved", "revision_requested"}:
        raise HTTPException(status_code=400, detail="status must be 'approved' or 'revision_requested'")
    record = repository.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    repository.update_human_review(generation_id, status=req.status, comment=req.comment)
    return repository.get_generation(generation_id)


@app.get("/generations", response_model=list[GenerationRecord])
def list_generations(property_id: str | None = None) -> list[GenerationRecord]:
    return repository.list_generations(property_id)


# ---------------------------------------------------------------------------
# Blog Generation
# ---------------------------------------------------------------------------

@app.post("/generate/blog", response_model=BlogGenerationRecord)
def generate_blog(req: GenerateBlogRequest) -> BlogGenerationRecord:
    try:
        blog_dir = brief_dir(settings.data_dir_resolved / "blogs", req.blog_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not blog_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown blog_id '{req.blog_id}'")
    return run_blog_workflow(blog_dir)


@app.get("/blogs/{generation_id}", response_model=BlogGenerationRecord)
def get_blog_generation(generation_id: str) -> BlogGenerationRecord:
    record = blog_repository.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Blog generation not found")
    return record


@app.post("/blogs/{generation_id}/review", response_model=BlogGenerationRecord)
def review_blog_generation(generation_id: str, req: HumanReviewRequest) -> BlogGenerationRecord:
    if req.status not in {"approved", "revision_requested"}:
        raise HTTPException(status_code=400, detail="status must be 'approved' or 'revision_requested'")
    record = blog_repository.get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Blog generation not found")
    blog_repository.update_human_review(generation_id, status=req.status, comment=req.comment)
    return blog_repository.get_generation(generation_id)


@app.get("/blogs", response_model=list[BlogGenerationRecord])
def list_blog_generations(blog_id: str | None = None) -> list[BlogGenerationRecord]:
    return blog_repository.list_generations(blog_id)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
