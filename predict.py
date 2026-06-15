"""
SIA Inference Script — predict.py
Accepts CSV input or single-ticket dict; outputs predictions + Evidence Dossiers.
"""

import argparse, json, joblib, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

MODEL_DIR   = "models/"
RESULTS_DIR = "results/"


def load_models():
    clf_bundle = joblib.load(f"{MODEL_DIR}/classifier.pkl")
    encoders   = joblib.load(f"{MODEL_DIR}/feature_encoders.pkl")
    clustering = joblib.load(f"{MODEL_DIR}/clustering.pkl")
    return clf_bundle, encoders, clustering


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {
        "ticket_subject":     ["ticket_subject", "subject"],
        "ticket_description": ["ticket_description", "description", "body"],
        "ticket_priority":    ["ticket_priority", "priority"],
        "ticket_channel":     ["ticket_channel", "channel"],
        "ticket_type":        ["ticket_type", "type", "category"],
        "customer_email":     ["customer_email", "email"],
        "product_purchased":  ["product_purchased", "product"],
        "resolution_time":    ["resolution_time", "time_to_resolve",
                               "resolution_time_(in_hours)", "resolution_time_hours"],
    }
    rename = {}
    for canonical, aliases in col_map.items():
        for alias in aliases:
            if alias in df.columns and canonical not in df.columns:
                rename[alias] = canonical
    df.rename(columns=rename, inplace=True)

    for col in ["ticket_subject", "ticket_description"]:
        if col not in df.columns:
            df[col] = ""
    df["full_text"] = (df["ticket_subject"].fillna("") + " " +
                       df["ticket_description"].fillna("")).str.strip()

    for col in ["ticket_channel", "ticket_type", "product_purchased", "customer_email"]:
        if col not in df.columns:
            df[col] = "unknown"

    if "resolution_time" not in df.columns:
        df["resolution_time"] = 24.0
    df["resolution_time"] = pd.to_numeric(df["resolution_time"], errors="coerce").fillna(24.0)

    if "ticket_priority" not in df.columns:
        df["ticket_priority"] = "Medium"
    priority_map = {"low": "Low", "medium": "Medium", "high": "High", "critical": "Critical"}
    df["ticket_priority"] = (
        df["ticket_priority"].astype(str).str.strip().str.lower()
        .map(priority_map).fillna("Medium")
    )

    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]

    return df


def compute_signals(df: pd.DataFrame, clustering: dict) -> pd.DataFrame:
    from train_pipeline import (
        nlp_signal, resolution_time_signal, channel_signal,
        PRIORITY_TO_NUM, NUM_TO_PRIORITY
    )

    df["sig1_nlp"]     = nlp_signal(df)
    df["sig3_restime"] = resolution_time_signal(df)
    df["sig4_channel"] = channel_signal(df)

    # Embedding signal
    sbert  = SentenceTransformer(clustering["sbert_name"])
    embs   = sbert.encode(df["full_text"].tolist(), batch_size=64,
                           show_progress_bar=False, normalize_embeddings=True)
    labels = clustering["kmeans"].predict(embs)
    df["sig2_embed"] = np.array([clustering["rank_map"][int(l)] for l in labels])

    df["fused_score"]      = (0.40*df["sig1_nlp"] + 0.35*df["sig2_embed"] +
                               0.15*df["sig3_restime"] + 0.10*df["sig4_channel"])
    df["inferred_num"]     = np.clip(np.round(df["fused_score"]).astype(int), 0, 3)
    df["inferred_severity"] = df["inferred_num"].map(NUM_TO_PRIORITY)
    df["assigned_num"]     = df["ticket_priority"].map(PRIORITY_TO_NUM).fillna(1).astype(int)
    df["severity_delta"]   = df["inferred_num"] - df["assigned_num"]
    df["mismatch_type"]    = df.apply(
        lambda r: "Hidden Crisis" if r["severity_delta"] > 0
        else ("False Alarm" if r["severity_delta"] < 0 else "Consistent"), axis=1
    )
    return df


def build_features(df: pd.DataFrame, encoders: dict) -> object:
    tfidf   = encoders["tfidf"]
    le_chan  = encoders["le_channel"]
    le_type  = encoders["le_type"]
    le_prod  = encoders["le_product"]
    le_dom   = encoders["le_domain"]
    scaler   = encoders["scaler"]
    cols     = encoders["structured_cols"]

    X_tfidf = tfidf.transform(df["full_text"])

    def safe_enc(le, vals):
        classes = set(le.classes_)
        return np.array([le.transform([v])[0] if v in classes else 0 for v in vals])

    df = df.copy()
    df["channel_enc"] = safe_enc(le_chan, df["ticket_channel"].astype(str))
    df["type_enc"]    = safe_enc(le_type,  df["ticket_type"].astype(str))
    df["product_enc"] = safe_enc(le_prod,  df["product_purchased"].astype(str))
    df["email_domain"] = df["customer_email"].astype(str).apply(
        lambda e: e.split("@")[-1].split(".")[0] if "@" in str(e) else "unknown"
    )
    df["domain_enc"]       = safe_enc(le_dom, df["email_domain"].astype(str))
    df["nlp_x_embed"]      = df["sig1_nlp"] * df["sig2_embed"]
    df["nlp_x_restime"]    = df["sig1_nlp"] * df["sig3_restime"]
    df["delta_abs"]         = df["severity_delta"].abs()
    df["both_high"]         = ((df["sig1_nlp"] >= 2) & (df["sig2_embed"] >= 2)).astype(int)
    df["both_low"]          = ((df["sig1_nlp"] <= 0.5) & (df["sig2_embed"] <= 0.5)).astype(int)
    df["assigned_num_feat"] = df["assigned_num"].astype(float)
    df["inferred_num_feat"] = df["inferred_num"].astype(float)

    X_struct = scaler.transform(df[cols].fillna(0))
    return hstack([X_tfidf, csr_matrix(X_struct)]), df


def ensemble_predict(X, clf_bundle: dict):
    lr, rf, gbm = clf_bundle["lr"], clf_bundle["rf"], clf_bundle["gbm"]
    w1, w2, w3  = clf_bundle["weights"]
    threshold   = clf_bundle["threshold"]
    n_struct    = clf_bundle["n_struct"]

    lr_p  = lr.predict_proba(X)[:, 1]
    rf_p  = rf.predict_proba(X)[:, 1]
    gbm_p = gbm.predict_proba(X[:, -n_struct:].toarray())[:, 1]
    proba = w1*lr_p + w2*rf_p + w3*gbm_p
    preds = (proba >= threshold).astype(int)
    return preds, proba


def predict(input_path: str, output_path: str):
    from train_pipeline import generate_dossier

    print(f"[PREDICT] Input: {input_path}")
    raw = pd.read_csv(input_path)
    df  = preprocess_input(raw)

    clf_bundle, encoders, clustering = load_models()
    df = compute_signals(df, clustering)
    X, df = build_features(df, encoders)
    preds, proba = ensemble_predict(X, clf_bundle)

    df["mismatch_label"] = preds
    df["mismatch_prob"]  = proba

    dossiers = [
        generate_dossier(row)
        for _, row in tqdm(df[df["mismatch_label"] == 1].iterrows(),
                           total=int(preds.sum()), desc="[DOSSIER]")
    ]

    df.to_csv(output_path, index=False)
    dp = output_path.replace(".csv", "_dossiers.json")
    with open(dp, "w") as fh:
        json.dump(dossiers, fh, indent=2)

    print(f"[PREDICT] Mismatch rate: {preds.mean():.2%} | Flagged: {int(preds.sum())}")
    print(f"[PREDICT] Saved → {output_path}")
    return df, dossiers


def predict_single(ticket: dict) -> dict:
    from train_pipeline import generate_dossier

    clf_bundle, encoders, clustering = load_models()
    df = preprocess_input(pd.DataFrame([ticket]))
    df = compute_signals(df, clustering)
    X, df = build_features(df, encoders)

    preds, proba = ensemble_predict(X, clf_bundle)
    pred = int(preds[0])
    prob = float(proba[0])

    row = df.iloc[0].copy()
    row["mismatch_label"] = pred
    row["mismatch_prob"]  = prob

    return {
        "mismatch_label":    pred,
        "mismatch_prob":     prob,
        "inferred_severity": str(df["inferred_severity"].iloc[0]),
        "mismatch_type":     str(df["mismatch_type"].iloc[0]),
        "fused_score":       float(df["fused_score"].iloc[0]),
        "dossier":           generate_dossier(row) if pred == 1 else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/tickets.csv")
    parser.add_argument("--output", default="results/predictions.csv")
    args = parser.parse_args()
    predict(args.input, args.output)