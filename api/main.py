"""
FastAPI backend for the Instacart recommendation + RAG pipeline.

Endpoints
---------
GET  /health        – liveness / readiness check
POST /recommend     – full pipeline: two-tower retrieval → inventory constraints → RAG policy reasoning
POST /recommend/fast – retrieval-only (no RAG), returns top-k items quickly
"""

import json
import os
import re
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── ensure project root is on sys.path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Pydantic schemas ──────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="Instacart user_id")
    intent: str = Field(
        default="Weekly grocery restock for a family of four",
        description="Natural-language shopping intent for the RAG agent",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class FastRecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="Instacart user_id")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class RecommendItem(BaseModel):
    product_id: int
    product_name: str
    aisle: str = ""
    department: str = ""
    score: float = 0.0
    stock_status: str = ""
    policy_notes: str = ""


class RecommendResponse(BaseModel):
    user_id: int
    intent: str
    recommendations: list[RecommendItem]
    substitutions: dict = {}
    warnings: list[str] = []
    citations: list[str] = []
    answer_summary: str = ""
    fallback_used: bool = False
    telemetry_ms: dict = {}


class FastRecommendResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendItem]
    fallback_used: bool = False


class HealthResponse(BaseModel):
    status: str
    model_version: str = "not_loaded"
    model_loaded: bool = False


# ── app state (populated at startup) ──────────────────────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up: load model + build FAISS index once at startup."""
    print("[api] Warming up model …")
    try:
        from rag_agent.graph import _load_model_components
        cache = _load_model_components()
        _state["model_loaded"] = True
        _state["model_version"] = cache.get("model_version", "unknown")
        print(f"[api] Model ready — {_state['model_version']}")
    except Exception as e:
        print(f"[api] Model warmup failed: {e}", file=sys.stderr)
        _state["model_loaded"] = False
        _state["model_version"] = "not_loaded"
    yield
    _state.clear()


app = FastAPI(
    title="Instacart RecSys API",
    version="1.0.0",
    description="Two-tower retrieval + RAG policy-compliance pipeline",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── health ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_version=_state.get("model_version", "not_loaded"),
        model_loaded=_state.get("model_loaded", False),
    )


# ── full pipeline ─────────────────────────────────────────────────────────

def _parse_answer_json(raw_answer: str) -> dict:
    """Best-effort parse of the LLM JSON answer."""
    try:
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw_answer.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """Run the full RAG pipeline: retrieval → constraints → policy → generation."""
    try:
        from rag_agent.graph import run_pipeline

        result = run_pipeline(user_id=req.user_id, intent=req.intent)

        # Build structured recommendation items
        items: list[RecommendItem] = []
        stock_map = result.get("stock_map", {})

        # Try to parse structured answer from LLM
        answer_json = _parse_answer_json(result.get("answer", ""))
        answer_items = {
            it.get("product_id"): it
            for it in answer_json.get("recommended_items", [])
        }

        for rec in result.get("final_recommendations", [])[:req.top_k]:
            pid = rec["product_id"]
            ai = answer_items.get(pid, {})
            items.append(RecommendItem(
                product_id=pid,
                product_name=rec.get("product_name", f"Product {pid}"),
                aisle=rec.get("aisle", ""),
                department=rec.get("department", ""),
                score=rec.get("score", 0.0),
                stock_status=stock_map.get(pid, stock_map.get(str(pid), "")),
                policy_notes=ai.get("policy_notes", ai.get("reason", "")),
            ))

        # Extract summary
        summary = answer_json.get("summary", "")
        if not summary and result.get("answer"):
            summary = result["answer"][:500]

        return RecommendResponse(
            user_id=result["user_id"],
            intent=result["intent"],
            recommendations=items,
            substitutions=result.get("substitutions", {}),
            warnings=result.get("warnings", []),
            citations=result.get("citations", []),
            answer_summary=summary,
            fallback_used=result.get("fallback_used", False),
            telemetry_ms=result.get("telemetry_ms", {}),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── fast retrieval-only ───────────────────────────────────────────────────

@app.post("/recommend/fast", response_model=FastRecommendResponse)
async def recommend_fast(req: FastRecommendRequest):
    """Retrieval-only: two-tower model → FAISS → top-k items (no RAG)."""
    try:
        from rag_agent.graph import get_recs_for_user

        recs, fallback_used = get_recs_for_user(req.user_id, k=req.top_k)

        items = [
            RecommendItem(
                product_id=r["product_id"],
                product_name=r.get("product_name", f"Product {r['product_id']}"),
                aisle=r.get("aisle", ""),
                department=r.get("department", ""),
                score=r.get("score", 0.0),
            )
            for r in recs[:req.top_k]
        ]

        return FastRecommendResponse(
            user_id=req.user_id,
            recommendations=items,
            fallback_used=fallback_used,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
