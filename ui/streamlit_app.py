"""
Invoice Intelligence System — Streamlit UI
Presentation-ready interface with ML visualizations.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Invoice Intelligence System", page_icon="🧠", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%); }
.stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%); }
h1, h2, h3 { color: #e0e0ff !important; }
.glass-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 1.5rem; margin: 0.5rem 0;
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.metric-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3); border-radius: 12px;
    padding: 1.2rem; text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #667eea; }
.metric-label { font-size: 0.85rem; color: #a0a0c0; margin-top: 0.3rem; }
.anomaly-tag {
    background: linear-gradient(135deg, #ff4757, #ff6b81); color: white;
    padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem;
}
.normal-tag {
    background: linear-gradient(135deg, #2ed573, #7bed9f); color: #1a1a2e;
    padding: 0.3rem 0.8rem; border-radius: 20px; font-weight: 600; font-size: 0.8rem;
}
.hero-title {
    font-size: 2.5rem; font-weight: 700; text-align: center;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-sub { text-align: center; color: #a0a0c0; font-size: 1.1rem; margin-bottom: 2rem; }
.pipeline-step {
    display: inline-block; padding: 0.5rem 1rem; margin: 0.2rem;
    border-radius: 8px; font-size: 0.85rem; font-weight: 500;
}
div[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 1rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05); border-radius: 8px;
    color: #a0a0c0; padding: 0.5rem 1.5rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important;
}
</style>""", unsafe_allow_html=True)

# ── Plotly Theme ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e0e0ff"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    margin=dict(l=40, r=40, t=50, b=40),
)
CLUSTER_COLORS = px.colors.qualitative.Set2

def load_sample_data():
    p = Path(__file__).parent.parent / "data" / "sample_invoices.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

def run_ml_pipeline(entities_batch):
    """Run the ML pipeline (embeddings -> clustering -> anomaly) on entity data."""
    from app.services.embedding_service import EmbeddingService
    from app.services.clustering_service import ClusteringService
    from app.services.anomaly_service import AnomalyService

    if "embedding_service" not in st.session_state:
        with st.spinner("🔄 Loading ML models (first time only)..."):
            st.session_state.embedding_service = EmbeddingService()
            st.session_state.clustering_service = ClusteringService(eps=0.3, min_samples=3)
            st.session_state.anomaly_service = AnomalyService(contamination=0.15)

    emb_svc = st.session_state.embedding_service
    clust_svc = st.session_state.clustering_service
    anom_svc = st.session_state.anomaly_service

    embeddings = emb_svc.embed_invoice_entities(entities_batch)
    coords_2d = emb_svc.reduce_to_2d(embeddings, method="tsne")
    sim_matrix = emb_svc.compute_similarity_matrix(embeddings)
    clustering = clust_svc.fit_predict(embeddings)
    anomaly = anom_svc.fit_predict(embeddings)

    return embeddings, coords_2d, sim_matrix, clustering, anomaly

# ── HEADER ───────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 Invoice Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Unsupervised ML Pipeline: OCR → NER → Embeddings → DBSCAN Clustering → Isolation Forest Anomaly Detection</div>', unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📊 ML Visualizations", "🔬 Algorithm Explorer", "📋 Results Table", "📄 Upload Invoice"])

# ════════════════════ TAB 1: DASHBOARD ════════════════════
with tab1:
    st.markdown("### Pipeline Architecture")
    steps = ["📷 OCR", "🏷️ NER", "🔢 Embeddings", "📍 DBSCAN", "🚨 Isolation Forest"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        col.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.5rem;">Step {i+1}</div><div class="metric-label">{step}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🚀 Load Sample Data & Run ML Pipeline", type="primary", use_container_width=True):
        data = load_sample_data()
        if data:
            with st.spinner("⏳ Running Unsupervised ML Pipeline..."):
                emb, coords, sim, clust, anom = run_ml_pipeline(data)
                st.session_state.update({"sample_data": data, "embeddings": emb, "coords_2d": coords,
                    "sim_matrix": sim, "clustering": clust, "anomaly": anom, "pipeline_done": True})
            st.success(f"✅ Pipeline complete! Processed {len(data)} invoices.")
        else:
            st.error("Sample data not found.")

    if st.session_state.get("pipeline_done"):
        clust = st.session_state.clustering
        anom = st.session_state.anomaly
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.sample_data)}</div><div class="metric-label">Total Invoices</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{clust.n_clusters}</div><div class="metric-label">Clusters Found</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{anom.n_anomalies}</div><div class="metric-label">Anomalies Detected</div></div>', unsafe_allow_html=True)
        sil = f"{clust.silhouette_avg:.3f}" if clust.silhouette_avg else "N/A"
        c4.markdown(f'<div class="metric-card"><div class="metric-value">{sil}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)

# ════════════════════ TAB 2: ML VISUALIZATIONS ════════════════════
with tab2:
    if not st.session_state.get("pipeline_done"):
        st.info("👆 Go to Dashboard tab and click **Load Sample Data** first.")
    else:
        data = st.session_state.sample_data
        coords = st.session_state.coords_2d
        clust = st.session_state.clustering
        anom = st.session_state.anomaly
        sim = st.session_state.sim_matrix
        labels = clust.labels
        anom_labels = anom.labels

        vendors = [d.get("vendor_name", {}).get("value", "Unknown") or "Unknown" for d in data]
        inv_nums = [d.get("invoice_number", {}).get("value", "N/A") or "N/A" for d in data]

        # ── t-SNE Scatter Plot ───────────────────────────────────────
        st.markdown("### 🗺️ t-SNE Embedding Visualization")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Each point is an invoice. Colors = DBSCAN clusters. Red stars ⭐ = anomalies detected by Isolation Forest.")

        df_scatter = pd.DataFrame({
            "t-SNE 1": coords[:, 0], "t-SNE 2": coords[:, 1],
            "Cluster": [f"Cluster {l}" if l != -1 else "Noise" for l in labels],
            "Anomaly": ["⚠️ Anomaly" if a == -1 else "Normal" for a in anom_labels],
            "Vendor": vendors, "Invoice": inv_nums,
            "Score": [round(float(s), 4) for s in anom.scores],
        })

        fig = px.scatter(df_scatter, x="t-SNE 1", y="t-SNE 2", color="Cluster",
            symbol="Anomaly", hover_data=["Vendor", "Invoice", "Score"],
            color_discrete_sequence=CLUSTER_COLORS, title="t-SNE 2D Projection of Invoice Embeddings")
        fig.update_layout(**PLOTLY_LAYOUT, height=500)
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns(2)

        # ── Cluster Distribution ─────────────────────────────────────
        with col_left:
            st.markdown("### 📊 Cluster Size Distribution")
            cluster_names = [f"Cluster {k}" for k in clust.cluster_sizes.keys()] + (["Noise"] if clust.n_noise > 0 else [])
            cluster_vals = list(clust.cluster_sizes.values()) + ([clust.n_noise] if clust.n_noise > 0 else [])
            colors = list(CLUSTER_COLORS[:len(clust.cluster_sizes)]) + (["#ff4757"] if clust.n_noise > 0 else [])
            fig2 = go.Figure(go.Bar(x=cluster_names, y=cluster_vals, marker_color=colors, text=cluster_vals, textposition="auto"))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Invoices per Cluster", height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # ── Anomaly Score Distribution ───────────────────────────────
        with col_right:
            st.markdown("### 🚨 Anomaly Score Distribution")
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=anom.scores, nbinsx=15, marker_color="#667eea", opacity=0.8, name="Scores"))
            if anom.score_threshold:
                fig3.add_vline(x=anom.score_threshold, line_dash="dash", line_color="#ff4757",
                    annotation_text="Threshold", annotation_font_color="#ff4757")
            fig3.update_layout(**PLOTLY_LAYOUT, title="Isolation Forest Score Distribution", height=400,
                xaxis_title="Anomaly Score", yaxis_title="Count")
            st.plotly_chart(fig3, use_container_width=True)

        # ── Cosine Similarity Heatmap ────────────────────────────────
        st.markdown("### 🔥 Invoice Similarity Heatmap")
        st.caption("Cosine similarity between invoice embeddings. Bright = similar, Dark = different.")
        heatmap_labels = [f"{v[:12]}..{n}" for v, n in zip(vendors, inv_nums)]
        fig4 = go.Figure(go.Heatmap(z=sim, x=heatmap_labels, y=heatmap_labels,
            colorscale=[[0, "#0a0a1a"], [0.5, "#667eea"], [1, "#764ba2"]], zmin=0, zmax=1))
        fig4.update_layout(**PLOTLY_LAYOUT, height=600, title="Pairwise Cosine Similarity Matrix")
        fig4.update_xaxes(tickangle=45, tickfont=dict(size=8))
        fig4.update_yaxes(tickfont=dict(size=8))
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════ TAB 3: ALGORITHM EXPLORER ════════════════════
with tab3:
    if not st.session_state.get("pipeline_done"):
        st.info("👆 Go to Dashboard tab and click **Load Sample Data** first.")
    else:
        emb = st.session_state.embeddings
        coords = st.session_state.coords_2d
        data = st.session_state.sample_data
        vendors = [d.get("vendor_name", {}).get("value", "Unknown") or "Unknown" for d in data]

        st.markdown("### 🔬 Interactive Parameter Tuning")
        st.markdown("Adjust the hyperparameters and watch how clustering & anomaly detection change in real-time.")

        col_dbscan, col_iforest = st.columns(2)

        with col_dbscan:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### DBSCAN Parameters")
            st.markdown("""
            - **eps (ε)**: Max distance for neighbors. Lower = tighter clusters.
            - **min_samples**: Min points to form a cluster core.
            """)
            eps_val = st.slider("Epsilon (ε)", 0.05, 1.0, 0.3, 0.05, key="eps_slider")
            min_samp = st.slider("Min Samples", 2, 8, 3, 1, key="ms_slider")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_iforest:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Isolation Forest Parameters")
            st.markdown("""
            - **contamination**: Expected proportion of anomalies.
            - Higher = more points flagged as anomalous.
            """)
            contam_val = st.slider("Contamination", 0.01, 0.5, 0.15, 0.01, key="contam_slider")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔄 Re-run with New Parameters", type="primary", use_container_width=True):
            from app.services.clustering_service import ClusteringService
            from app.services.anomaly_service import AnomalyService

            new_clust = ClusteringService(eps=eps_val, min_samples=min_samp).fit_predict(emb)
            new_anom = AnomalyService(contamination=contam_val).fit_predict(emb)

            st.session_state.explorer_clust = new_clust
            st.session_state.explorer_anom = new_anom

        if "explorer_clust" in st.session_state:
            ec = st.session_state.explorer_clust
            ea = st.session_state.explorer_anom

            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters", ec.n_clusters)
            m2.metric("Noise Points", ec.n_noise)
            m3.metric("Anomalies", ea.n_anomalies)

            df_exp = pd.DataFrame({
                "t-SNE 1": coords[:, 0], "t-SNE 2": coords[:, 1],
                "Cluster": [f"Cluster {l}" if l != -1 else "Noise" for l in ec.labels],
                "Anomaly": ["Anomaly" if a == -1 else "Normal" for a in ea.labels],
                "Vendor": vendors,
            })
            fig_exp = px.scatter(df_exp, x="t-SNE 1", y="t-SNE 2", color="Cluster", symbol="Anomaly",
                color_discrete_sequence=CLUSTER_COLORS,
                title=f"DBSCAN (ε={eps_val}, min_samples={min_samp}) + IForest (contamination={contam_val})")
            fig_exp.update_layout(**PLOTLY_LAYOUT, height=500)
            fig_exp.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
            st.plotly_chart(fig_exp, use_container_width=True)

# ════════════════════ TAB 4: RESULTS TABLE ════════════════════
with tab4:
    if not st.session_state.get("pipeline_done"):
        st.info("👆 Go to Dashboard tab and click **Load Sample Data** first.")
    else:
        data = st.session_state.sample_data
        clust = st.session_state.clustering
        anom = st.session_state.anomaly

        rows = []
        for i, d in enumerate(data):
            rows.append({
                "Invoice": d.get("invoice_number", {}).get("value") or "N/A",
                "Vendor": d.get("vendor_name", {}).get("value") or "Unknown",
                "Date": d.get("date", {}).get("value") or "N/A",
                "Amount": d.get("total_amount", {}).get("value") or "N/A",
                "Cluster": int(clust.labels[i]),
                "Anomaly Score": round(float(anom.scores[i]), 4),
                "Status": "⚠️ ANOMALY" if anom.labels[i] == -1 else "✅ Normal",
            })

        df = pd.DataFrame(rows)
        st.markdown("### 📋 Complete Results")

        # Highlight anomalies
        def highlight_anomalies(row):
            if "ANOMALY" in str(row["Status"]):
                return ["background-color: rgba(255,71,87,0.2)"] * len(row)
            return [""] * len(row)

        st.dataframe(df.style.apply(highlight_anomalies, axis=1), use_container_width=True, height=600)

        st.download_button("📥 Download as CSV", df.to_csv(index=False), "invoice_results.csv", "text/csv", use_container_width=True)
        st.download_button("📥 Download as JSON", json.dumps(rows, indent=2), "invoice_results.json", "application/json", use_container_width=True)

# ════════════════════ TAB 5: UPLOAD ════════════════════
with tab5:
    st.markdown("### 📄 Process a Real Invoice")
    st.markdown("Upload a PDF or image to run it through the full pipeline (requires OCR backend).")

    uploaded = st.file_uploader("Choose an invoice file", type=["pdf", "png", "jpg", "jpeg", "tiff"])
    if uploaded and st.button("🚀 Process Invoice", type="primary"):
        import requests
        api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        with st.spinner("Processing..."):
            try:
                resp = requests.post(f"{api_url}/process-invoice",
                    files={"file": (uploaded.name, uploaded.getbuffer(), uploaded.type)}, timeout=120)
                if resp.status_code == 200:
                    st.json(resp.json())
                else:
                    st.error(f"API Error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at {api_url}. Start the API server first.")

# ── Footer ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center; color:#606080; font-size:0.85rem;">
<p>Invoice Intelligence System v2.0 | Unsupervised ML: DBSCAN + Isolation Forest</p>
<p>Built with FastAPI, Streamlit, Sentence-Transformers, scikit-learn</p>
</div>""", unsafe_allow_html=True)
