"""
Streamlit UI for the Instacart RecSys + RAG pipeline.

Calls the FastAPI backend at http://localhost:8000 by default.
Launch:  streamlit run streamlit_app.py
"""

import json
import requests
import streamlit as st

st.set_page_config(
    page_title="Instacart RecSys Demo",
    page_icon="🛒",
    layout="wide",
)

# ── config ─────────────────────────────────────────────────────────────────
API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")

# ── sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Instacart RecSys")
st.sidebar.markdown("Two-tower retrieval + RAG policy agent")

# Check backend health
try:
    health = requests.get(f"{API_BASE}/health", timeout=3).json()
    model_ok = health.get("model_loaded", False)
    version = health.get("model_version", "unknown")
    if model_ok:
        st.sidebar.success(f"Model: {version}")
    else:
        st.sidebar.warning("Model not loaded")
except Exception:
    st.sidebar.error("API unreachable")
    health = {}

st.sidebar.divider()

# Demo users
DEMO_USERS = [1, 2, 6, 10, 138, 1550, 3]
user_id = st.sidebar.number_input("User ID", min_value=1, max_value=300_000, value=1, step=1)
st.sidebar.caption("Try demo users: " + ", ".join(str(u) for u in DEMO_USERS))

INTENT_PRESETS = {
    "Weekly restock": "Weekly grocery restock for a family of four",
    "Healthy snacks": "Healthy snack options, low sugar, high protein",
    "Quick weeknight dinners": "Quick weeknight dinner ingredients, under 30 minutes",
    "Baby supplies": "Essentials for a household with a newborn baby",
    "Party planning": "Snacks and drinks for a weekend get-together with friends",
    "Custom": "",
}

preset = st.sidebar.selectbox("Intent preset", list(INTENT_PRESETS.keys()))
if preset == "Custom":
    intent = st.sidebar.text_area("Intent", value="", height=80)
else:
    intent = st.sidebar.text_area("Intent", value=INTENT_PRESETS[preset], height=80)

top_k = st.sidebar.slider("Top-K", min_value=5, max_value=50, value=10)

use_rag = st.sidebar.toggle("Enable RAG policy agent", value=True)

run = st.sidebar.button("Get Recommendations", type="primary", use_container_width=True)

# ── main area ──────────────────────────────────────────────────────────────
st.title("🛒 Instacart Recommendation Engine")
st.caption("Two-Tower Retrieval + RAG Policy-Compliance Agent")

if not run:
    # Landing page
    st.markdown("""
    ### How it works
    1. **Two-Tower Model** retrieves personalized product candidates via FAISS ANN search
    2. **Inventory Constraints** filter out-of-stock items and suggest substitutions
    3. **RAG Policy Agent** grounds recommendations in store policy documents (GPT-4o)
    4. **Structured Output** returns items with policy citations, substitutions, and warnings

    **Select a user and intent in the sidebar, then click *Get Recommendations*.**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "Two-Tower")
    col2.metric("RAG Model", "GPT-4o")
    col3.metric("Retrieval", "FAISS + BM25")
    st.stop()

# ── API call ───────────────────────────────────────────────────────────────
with st.spinner("Running pipeline…"):
    try:
        if use_rag:
            resp = requests.post(
                f"{API_BASE}/recommend",
                json={"user_id": user_id, "intent": intent, "top_k": top_k},
                timeout=120,
            )
        else:
            resp = requests.post(
                f"{API_BASE}/recommend/fast",
                json={"user_id": user_id, "top_k": top_k},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is `uvicorn api.main:app` running?")
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ── telemetry bar ──────────────────────────────────────────────────────────
tel = data.get("telemetry_ms", {})
if tel:
    cols = st.columns(4)
    cols[0].metric("Total time", f"{tel.get('total', 0):.0f} ms")
    cols[1].metric("Retrieval", f"{tel.get('load_recs', 0):.0f} ms")
    cols[2].metric("Constraints", f"{tel.get('apply_constraints', 0):.0f} ms")
    cols[3].metric("Generation", f"{tel.get('generate_answer', 0):.0f} ms")

if data.get("fallback_used"):
    st.warning("Synthetic fallback used — user not in training data or model unavailable.")

# ── recommendations table ──────────────────────────────────────────────────
st.subheader("Recommendations")

recs = data.get("recommendations", [])
if not recs:
    st.info("No recommendations returned.")
    st.stop()

for i, item in enumerate(recs, 1):
    stock = item.get("stock_status", "")
    badge = {"in_stock": "🟢", "low_stock": "🟡", "out_of_stock": "🔴"}.get(stock, "⚪")

    with st.container():
        c1, c2, c3, c4 = st.columns([0.5, 4, 2, 1.5])
        c1.markdown(f"**{i}**")
        c2.markdown(f"**{item['product_name']}**")
        c3.caption(f"{item.get('aisle', '')} · {item.get('department', '')}")
        c4.markdown(f"{badge} `{item.get('score', 0):.4f}`")

        notes = item.get("policy_notes", "")
        if notes:
            with st.expander("Policy reasoning"):
                st.markdown(notes)

# ── substitutions ──────────────────────────────────────────────────────────
subs = data.get("substitutions", {})
if subs:
    st.subheader("Substitutions")
    for oos, sub in subs.items():
        if sub:
            st.markdown(f"- OOS product **{oos}** → substituted with **{sub}**")
        else:
            st.markdown(f"- OOS product **{oos}** → no substitute found")

# ── warnings ───────────────────────────────────────────────────────────────
warnings = data.get("warnings", [])
if warnings:
    st.subheader("Warnings")
    for w in warnings:
        st.warning(w)

# ── citations ──────────────────────────────────────────────────────────────
citations = data.get("citations", [])
if citations:
    st.subheader("Policy Citations")
    st.markdown(" · ".join(f"`{c}`" for c in citations))

# ── answer summary ─────────────────────────────────────────────────────────
summary = data.get("answer_summary", "")
if summary:
    with st.expander("Full LLM Answer"):
        st.markdown(summary)

# ── raw JSON viewer ────────────────────────────────────────────────────────
with st.expander("Raw API Response"):
    st.json(data)
