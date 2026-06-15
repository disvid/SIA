"""
SIA Inference Script
Usage:
  python predict.py --input data/enhanced_customer_support_data.csv --output results/predictions.csv
  python predict.py --input data/tickets.csv --output results/predictions.csv
"""

import argparse
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_DIR   = "models/"
RESULTS_DIR = "results/"
Path(RESULTS_DIR).mkdir(exist_ok=True)


def load_models():
    clf        = joblib.load(f"{MODEL_DIR}/classifier.pkl")
    encoders   = joblib.load(f"{MODEL_DIR}/feature_encoders.pkl")
    clustering = joblib.load(f"{MODEL_DIR}/clustering.pkl")
    return clf, encoders, clustering


def embed_and_cluster(texts: list, clustering: dict) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(clustering["sbert_name"])
    embeddings = sbert.encode(
        texts, batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    labels   = clustering["kmeans"].predict(embeddings)
    rank_map = clustering["rank_map"]
    default  = np.mean(list(rank_map.values()))
    return np.array([rank_map.get(int(l), default) for l in labels])


def safe_encode(le, values):
    """Encode unseen labels as 0 instead of crashing."""
    known = set(le.classes_)
    return np.array([le.transform([v])[0] if v in known else 0 for v in values])


def build_features(df: pd.DataFrame, encoders: dict) -> object:
    tfidf  = encoders["tfidf"]
    scaler = encoders["scaler"]
    cols   = encoders["structured_cols"]

    X_tfidf = tfidf.transform(df["full_text"])

    df["channel_enc"] = safe_encode(encoders["le_channel"], df["ticket_channel"].astype(str))
    df["type_enc"]    = safe_encode(encoders["le_type"],    df["ticket_type"].astype(str))
    df["agent_enc"]   = safe_encode(encoders["le_agent"],
                                    df.get("assigned_agent",
                                           pd.Series(["unknown"]*len(df))).astype(str))
    df["email_domain"] = df["customer_email"].astype(str).apply(
        lambda e: e.split("@")[-1].split(".")[0] if "@" in str(e) else "unknown"
    )
    df["domain_enc"]  = safe_encode(encoders["le_domain"], df["email_domain"].astype(str))

    X_struct = scaler.transform(df[cols].fillna(0))
    return hstack([X_tfidf, csr_matrix(X_struct)])


def preprocess_for_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training."""
    from train_pipeline import (
        nlp_signal, resolution_time_signal, satisfaction_signal,
        PRIORITY_TO_NUM, num_to_severity_label, CATEGORY_SEVERITY
    )

    # Column rename
    rename_map = {
        "Ticket_ID":             "ticket_id",
        "Customer_Name":         "customer_name",
        "Customer_Email":        "customer_email",
        "Ticket_Subject":        "ticket_subject",
        "Ticket_Description":    "ticket_description",
        "Issue_Category":        "ticket_type",
        "Priority_Level":        "ticket_priority",
        "Ticket_Channel":        "ticket_channel",
        "Submission_Date":       "submission_date",
        "Resolution_Time_Hours": "resolution_time",
        "Assigned_Agent":        "assigned_agent",
        "Satisfaction_Score":    "satisfaction_score",
    }
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=rename_map, inplace=True)

    for col in ["ticket_channel","ticket_type","customer_email","assigned_agent"]:
        if col not in df.columns:
            df[col] = "unknown"
    if "satisfaction_score" not in df.columns:
        df["satisfaction_score"] = 3.0
    if "ticket_age_days" not in df.columns:
        df["ticket_age_days"] = 0

    df["resolution_time"] = pd.to_numeric(df["resolution_time"], errors="coerce").fillna(24.0)
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce").fillna(3.0)

    priority_map = {"low":"Low","medium":"Medium","high":"High","critical":"Critical","urgent":"Critical"}
    df["ticket_priority"] = df["ticket_priority"].astype(str).str.strip().str.lower().map(priority_map).fillna("Medium")
    df["ticket_type"]     = df["ticket_type"].astype(str).str.strip()
    df["full_text"]       = (df["ticket_subject"].fillna("") + " " + df["ticket_description"].fillna("")).str.strip()
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["ticket_id"] = df["ticket_id"].astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


def predict(input_path: str, output_path: str):
    from train_pipeline import (
        nlp_signal, resolution_time_signal, satisfaction_signal,
        PRIORITY_TO_NUM, num_to_severity_label, generate_dossier
    )

    print(f"[PREDICT] Loading: {input_path}")
    raw = pd.read_csv(input_path)
    df  = preprocess_for_predict(raw)

    clf, encoders, clustering = load_models()

    # Signals
    df["sig1_nlp"]     = nlp_signal(df)
    df["sig2_embed"]   = embed_and_cluster(df["full_text"].tolist(), clustering)
    df["sig3_restime"] = resolution_time_signal(df)
    df["sig4_satisf"]  = satisfaction_signal(df)
    df["fused_score"]  = (0.35*df["sig1_nlp"] + 0.30*df["sig2_embed"] +
                          0.20*df["sig3_restime"] + 0.15*df["sig4_satisf"])

    df["inferred_severity"] = df["fused_score"].apply(num_to_severity_label)
    df["inferred_num"]      = df["inferred_severity"].map(PRIORITY_TO_NUM)
    df["assigned_num"]      = df["ticket_priority"].map(PRIORITY_TO_NUM).fillna(1)
    df["severity_delta"]    = df["inferred_num"] - df["assigned_num"]
    df["mismatch_type"]     = df.apply(
        lambda r: "Hidden Crisis" if r["severity_delta"] > 0
        else ("False Alarm" if r["severity_delta"] < 0 else "Consistent"),
        axis=1
    )

    X     = build_features(df, encoders)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    df["mismatch_label"] = preds
    df["mismatch_prob"]  = probs

    # Dossiers
    dossiers = []
    for _, row in tqdm(df[df["mismatch_label"]==1].iterrows(),
                       total=int(df["mismatch_label"].sum()), desc="[DOSSIER]"):
        dossiers.append(generate_dossier(row))

    df.to_csv(output_path, index=False)
    dossier_path = output_path.replace(".csv", "_dossiers.json")
    with open(dossier_path, "w") as fh:
        json.dump(dossiers, fh, indent=2)

    print(f"[PREDICT] Saved predictions → {output_path}")
    print(f"[PREDICT] Saved dossiers    → {dossier_path}")
    print(f"[PREDICT] Mismatch rate: {df['mismatch_label'].mean():.2%}")
    return df, dossiers


def predict_single(ticket: dict) -> dict:
    """Predict a single ticket (as dict). Returns prediction + dossier."""
    from train_pipeline import (
        nlp_signal, resolution_time_signal, satisfaction_signal,
        PRIORITY_TO_NUM, num_to_severity_label, generate_dossier
    )

    clf, encoders, clustering = load_models()

    df = pd.DataFrame([ticket])
    df = preprocess_for_predict(df)

    df["sig1_nlp"]     = nlp_signal(df)
    df["sig2_embed"]   = embed_and_cluster(df["full_text"].tolist(), clustering)
    df["sig3_restime"] = resolution_time_signal(df)
    df["sig4_satisf"]  = satisfaction_signal(df)
    df["fused_score"]  = (0.35*df["sig1_nlp"] + 0.30*df["sig2_embed"] +
                          0.20*df["sig3_restime"] + 0.15*df["sig4_satisf"])

    df["inferred_severity"] = df["fused_score"].apply(num_to_severity_label)
    df["inferred_num"]      = df["inferred_severity"].map(PRIORITY_TO_NUM)
    df["assigned_num"]      = df["ticket_priority"].map(PRIORITY_TO_NUM).fillna(1)
    df["severity_delta"]    = df["inferred_num"] - df["assigned_num"]
    df["mismatch_type"]     = df.apply(
        lambda r: "Hidden Crisis" if r["severity_delta"] > 0
        else ("False Alarm" if r["severity_delta"] < 0 else "Consistent"),
        axis=1
    )

    X    = build_features(df, encoders)
    pred = int(clf.predict(X)[0])
    prob = float(clf.predict_proba(X)[0, 1])

    row = df.iloc[0].copy()
    row["mismatch_label"] = pred
    row["mismatch_prob"]  = prob

    dossier = generate_dossier(row) if pred == 1 else None
    return {
        "mismatch_label":    pred,
        "mismatch_prob":     prob,
        "inferred_severity": str(df["inferred_severity"].iloc[0]),
        "mismatch_type":     str(df["mismatch_type"].iloc[0]),
        "fused_score":       float(df["fused_score"].iloc[0]),
        "dossier":           dossier
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference")
    parser.add_argument("--input",  default="data/enhanced_customer_support_data.csv")
    parser.add_argument("--output", default="results/predictions.csv")
    args = parser.parse_args()
    predict(args.input, args.output)