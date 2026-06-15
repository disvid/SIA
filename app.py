"""
SIA — Streamlit Web Application v3
Compatible with the v3 ensemble model (lr + rf + gbm).
"""

import json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .dossier-box {
        background: #1e1e2e; color: #cdd6f4;
        border-radius: 12px; padding: 1.5rem;
        border-left: 4px solid #cba6f7; font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🔍 SIA Navigation")
    page = st.radio("", [
        "📊 Dashboard",
        "🎫 Single Ticket Analyzer",
        "📁 Batch CSV Upload",
        "📈 Model Metrics",
        "🛡️ Adversarial Tests"
    ])
    st.markdown("---")
    st.markdown("**Support Integrity Auditor v3**")


@st.cache_data
def load_results():
    results = {}
    paths = {
        "pseudo":      "results/pseudo_labeled.csv",
        "metrics":     "results/metrics.json",
        "dossiers":    "results/dossiers.json",
        "ablation":    "results/ablation.json",
        "adversarial": "results/adversarial.json"
    }
    for key, path in paths.items():
        if Path(path).exists():
            if path.endswith(".csv"):
                results[key] = pd.read_csv(path)
            else:
                with open(path) as f:
                    results[key] = json.load(f)
    return results


@st.cache_resource
def load_predictor():
    try:
        from predict import predict_single
        return predict_single
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


# ── PAGE: DASHBOARD ──────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown('<div class="main-header">🔍 Support Integrity Auditor</div>', unsafe_allow_html=True)
    st.markdown("*Semantics-driven priority mismatch detection for enterprise CRM systems*")
    st.markdown("---")

    data = load_results()
    if "pseudo" not in data:
        st.warning("⚠️ No training results found. Run `python train_pipeline.py` first.")
        return

    df = data["pseudo"]
    total         = len(df)
    mismatched    = int(df["mismatch_label"].sum()) if "mismatch_label" in df.columns else 0
    hidden_crisis = int((df["mismatch_type"] == "Hidden Crisis").sum()) if "mismatch_type" in df.columns else 0
    false_alarm   = int((df["mismatch_type"] == "False Alarm").sum()) if "mismatch_type" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Tickets", f"{total:,}")
    c2.metric("Mismatches Detected", f"{mismatched:,}", delta=f"{mismatched/total:.1%} rate")
    c3.metric("🚨 Hidden Crises", f"{hidden_crisis:,}", delta="Under-prioritized", delta_color="inverse")
    c4.metric("⚠️ False Alarms", f"{false_alarm:,}", delta="Over-prioritized")
    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        if "mismatch_type" in df.columns:
            tc = df["mismatch_type"].value_counts().reset_index()
            tc.columns = ["Mismatch Type", "Count"]
            fig = px.pie(tc, values="Count", names="Mismatch Type",
                         title="🎯 Mismatch Type Distribution",
                         color_discrete_map={"Hidden Crisis":"#f38ba8","False Alarm":"#fab387","Consistent":"#a6e3a1"})
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        if "ticket_priority" in df.columns and "inferred_severity" in df.columns:
            comp = pd.concat([
                df["ticket_priority"].value_counts().rename("Assigned"),
                df["inferred_severity"].value_counts().rename("Inferred")
            ], axis=1).fillna(0).reset_index()
            comp.columns = ["Priority", "Assigned", "Inferred"]
            fig2 = px.bar(comp.melt(id_vars="Priority"), x="Priority", y="value",
                          color="variable", barmode="group",
                          title="📊 Assigned vs Inferred Priority Distribution",
                          labels={"value":"Count","variable":"Type"},
                          color_discrete_sequence=["#89b4fa","#cba6f7"])
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🌡️ Severity Delta Heatmap")
    if all(c in df.columns for c in ["ticket_type","ticket_channel","severity_delta"]):
        hm = df.groupby(["ticket_type","ticket_channel"])["severity_delta"].mean().reset_index()
        pivot = hm.pivot(index="ticket_type", columns="ticket_channel", values="severity_delta").fillna(0)
        fig3 = px.imshow(pivot, color_continuous_scale="RdBu_r",
                         title="Mean Severity Delta (Inferred − Assigned) by Category × Channel",
                         labels={"color":"Delta"}, aspect="auto")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### 📡 Ablation Study")
    if "ablation" in data:
        abl_df = pd.DataFrame(data["ablation"]).T.reset_index()
        abl_df.columns = ["Signal Set","Accuracy","Macro F1"]
        fig4 = px.bar(abl_df.melt(id_vars="Signal Set"), x="Signal Set", y="value",
                      color="variable", barmode="group",
                      title="Ablation Study — Signal Contribution",
                      color_discrete_sequence=["#89b4fa","#a6e3a1"])
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 🗂️ Recently Flagged Tickets")
    if "mismatch_label" in df.columns:
        flagged = df[df["mismatch_label"]==1][
            ["ticket_id","ticket_priority","inferred_severity","mismatch_type","severity_delta","fused_score"]
        ].head(20)
        st.dataframe(flagged, use_container_width=True)


# ── PAGE: SINGLE TICKET ──────────────────────────────────────────────────────
def page_single():
    st.markdown("## 🎫 Single Ticket Analyzer")

    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        with c1:
            subject  = st.text_input("Ticket Subject", "System is completely down, cannot access")
            priority = st.selectbox("Assigned Priority", ["Low","Medium","High","Critical"])
            channel  = st.selectbox("Channel", ["email","chat","phone","social media"])
        with c2:
            ttype   = st.text_input("Ticket Type", "Technical Issue")
            product = st.text_input("Product Purchased", "Enterprise Suite")
            rt      = st.number_input("Resolution Time (hours)", min_value=0.0, value=48.0, step=1.0)

        desc = st.text_area("Ticket Description",
            "Our entire production environment is DOWN. All users cannot access the platform. "
            "This is causing massive revenue loss. We need IMMEDIATE help. Please escalate NOW.",
            height=150)
        email     = st.text_input("Customer Email", "admin@bigcorp.com")
        ticket_id = st.text_input("Ticket ID", "TKT-99999")
        submitted = st.form_submit_button("🔍 Analyze Ticket", use_container_width=True)

    if submitted:
        predict_single = load_predictor()
        if predict_single is None:
            st.error("❌ Model not loaded. Run `python train_pipeline.py` first.")
            return

        ticket = {
            "ticket_id": ticket_id, "ticket_subject": subject,
            "ticket_description": desc, "ticket_priority": priority,
            "ticket_channel": channel, "ticket_type": ttype,
            "product_purchased": product, "resolution_time": rt,
            "customer_email": email
        }

        with st.spinner("Analyzing ticket..."):
            result = predict_single(ticket)

        mismatch = result["mismatch_label"]
        prob     = result["mismatch_prob"]
        inferred = result["inferred_severity"]
        mtype    = result["mismatch_type"]

        if mismatch:
            if mtype == "Hidden Crisis":
                st.error(f"🚨 **MISMATCH DETECTED: Hidden Crisis** | Confidence: {prob:.2%}")
            else:
                st.warning(f"⚠️ **MISMATCH DETECTED: False Alarm** | Confidence: {prob:.2%}")
        else:
            st.success(f"✅ **No Mismatch Detected** | Confidence: {1-prob:.2%}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Assigned Priority", priority)
        c2.metric("Inferred Severity", inferred)
        c3.metric("Mismatch Probability", f"{prob:.2%}")

        if result.get("dossier"):
            st.markdown("### 📋 Evidence Dossier")
            st.markdown(f'<div class="dossier-box"><pre>{json.dumps(result["dossier"], indent=2)}</pre></div>',
                        unsafe_allow_html=True)


# ── PAGE: BATCH CSV ──────────────────────────────────────────────────────────
def page_batch():
    st.markdown("## 📁 Batch CSV Upload")
    uploaded = st.file_uploader("Upload tickets CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_raw)} tickets.")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🔍 Run Batch Analysis"):
            from predict import predict
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_in:
                df_raw.to_csv(tmp_in.name, index=False)
                tmp_input = tmp_in.name
            tmp_output = tmp_input.replace(".csv", "_out.csv")

            with st.spinner("Running batch analysis..."):
                df_out, dossiers = predict(tmp_input, tmp_output)

            st.success(f"Done! {int(df_out['mismatch_label'].sum())} mismatches detected.")
            st.dataframe(df_out[["ticket_id","ticket_priority","inferred_severity",
                                  "mismatch_type","mismatch_prob"]].head(50),
                         use_container_width=True)

            st.download_button("⬇️ Download Predictions CSV",
                               df_out.to_csv(index=False).encode(),
                               file_name="sia_predictions.csv")
            st.download_button("⬇️ Download Dossiers JSON",
                               json.dumps(dossiers, indent=2).encode(),
                               file_name="sia_dossiers.json")
            os.unlink(tmp_input)


# ── PAGE: METRICS ────────────────────────────────────────────────────────────
def page_metrics():
    st.markdown("## 📈 Model Metrics")
    data = load_results()

    if "metrics" in data:
        m = data["metrics"]
        THRESHOLDS = {"accuracy": 0.83, "macro_f1": 0.82,
                      "recall_consistent": 0.78, "recall_mismatch": 0.78}
        for key, thresh in THRESHOLDS.items():
            val = m.get(key, 0)
            label = key.replace("_", " ").title()
            delta = f"{val - thresh:+.4f} vs {thresh}"
            delta_color = "normal" if val >= thresh else "inverse"
            st.metric(label, f"{val:.4f}", delta=delta, delta_color=delta_color)

        all_pass = all(m.get(k, 0) >= v for k, v in THRESHOLDS.items())
        if all_pass:
            st.success("🎉 All verification thresholds MET!")
        else:
            failed = [k for k, v in THRESHOLDS.items() if m.get(k, 0) < v]
            st.error(f"❌ Failed: {', '.join(failed)}")
    else:
        st.warning("Run `python train_pipeline.py` first.")


# ── PAGE: ADVERSARIAL ────────────────────────────────────────────────────────
def page_adversarial():
    st.markdown("## 🛡️ Adversarial Test Results")
    data = load_results()

    if "adversarial" in data:
        adv = data["adversarial"]
        score = adv.get("score", 0)
        correct = adv.get("correct", 0)

        st.metric("Adversarial Score", f"{correct}/10 ({score:.0%})")
        if score >= 0.70:
            st.success("✅ BONUS THRESHOLD MET (≥7/10) — +10% score bonus!")
        else:
            st.warning(f"⚠️ Score {correct}/10 — need 7/10 for bonus.")

        results_df = pd.DataFrame(adv.get("results", []))
        if not results_df.empty:
            results_df["Status"] = results_df["correct"].map({True: "✅ Correct", False: "❌ Wrong"})
            st.dataframe(results_df[["ticket_id","predicted","expected","correct","probability","mismatch_type","Status"]],
                         use_container_width=True)
    else:
        st.warning("Run `python train_pipeline.py` first.")


# ── ROUTER ───────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    page_dashboard()
elif page == "🎫 Single Ticket Analyzer":
    page_single()
elif page == "📁 Batch CSV Upload":
    page_batch()
elif page == "📈 Model Metrics":
    page_metrics()
elif page == "🛡️ Adversarial Tests":
    page_adversarial()