"""
SIA Training Pipeline — Adapted for enhanced_customer_support_data.csv / tickets.csv
Columns: Ticket_ID, Customer_Name, Customer_Email, Ticket_Subject, Ticket_Description,
         Issue_Category, Priority_Level, Ticket_Channel, Submission_Date,
         Resolution_Time_Hours, Assigned_Agent, Satisfaction_Score
"""

import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, classification_report
)
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH        = "data/enhanced_customer_support_data.csv"   # primary
FALLBACK_PATH    = "data/tickets.csv"                          # fallback
MODEL_DIR        = "models/"
RESULTS_DIR      = "results/"
Path(MODEL_DIR).mkdir(exist_ok=True)
Path(RESULTS_DIR).mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path: str = None) -> pd.DataFrame:
    # Auto-select file
    if path is None or not Path(path).exists():
        if Path(DATA_PATH).exists():
            path = DATA_PATH
        elif Path(FALLBACK_PATH).exists():
            path = FALLBACK_PATH
        else:
            raise FileNotFoundError(
                "Neither enhanced_customer_support_data.csv nor tickets.csv found in data/ folder."
            )

    print(f"[DATA] Loading: {path}")
    df = pd.read_csv(path)

    # ── Standardize column names: strip spaces, lowercase
    df.columns = [c.strip() for c in df.columns]

    # ── Exact column mapping for YOUR dataset
    rename_map = {
        "Ticket_ID":            "ticket_id",
        "Customer_Name":        "customer_name",
        "Customer_Email":       "customer_email",
        "Ticket_Subject":       "ticket_subject",
        "Ticket_Description":   "ticket_description",
        "Issue_Category":       "ticket_type",          # maps to ticket_type
        "Priority_Level":       "ticket_priority",      # maps to ticket_priority
        "Ticket_Channel":       "ticket_channel",
        "Submission_Date":      "submission_date",
        "Resolution_Time_Hours":"resolution_time",      # maps to resolution_time
        "Assigned_Agent":       "assigned_agent",
        "Satisfaction_Score":   "satisfaction_score",
    }
    df.rename(columns=rename_map, inplace=True)

    # ── Fill any columns that might still be missing
    for col in ["ticket_channel", "ticket_type", "customer_email", "assigned_agent"]:
        if col not in df.columns:
            df[col] = "unknown"

    # ── Resolution time
    df["resolution_time"] = pd.to_numeric(df["resolution_time"], errors="coerce")
    median_rt = df["resolution_time"].median()
    df["resolution_time"].fillna(median_rt if not np.isnan(median_rt) else 24.0,
                                 inplace=True)

    # ── Satisfaction score (extra signal available in this dataset)
    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")
        df["satisfaction_score"].fillna(df["satisfaction_score"].median(), inplace=True)
    else:
        df["satisfaction_score"] = 3.0

    # ── Priority normalization — your dataset uses: High, Medium, Low (no Critical)
    priority_map = {
        "low":      "Low",
        "medium":   "Medium",
        "med":      "Medium",
        "high":     "High",
        "critical": "Critical",
        "urgent":   "Critical",
    }
    df["ticket_priority"] = (
        df["ticket_priority"]
        .astype(str).str.strip().str.lower()
        .map(priority_map)
        .fillna("Medium")
    )

    # ── Ticket type normalization
    df["ticket_type"] = df["ticket_type"].astype(str).str.strip()

    # ── Combine subject + description into full_text
    df["full_text"] = (
        df["ticket_subject"].fillna("") + " " +
        df["ticket_description"].fillna("")
    ).str.strip()

    # ── Ticket ID
    if "ticket_id" not in df.columns:
        df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
    df["ticket_id"] = df["ticket_id"].astype(str)

    # ── Parse submission date for seasonality/age feature
    if "submission_date" in df.columns:
        df["submission_date"] = pd.to_datetime(df["submission_date"], errors="coerce")
        df["ticket_age_days"] = (
            pd.Timestamp.now() - df["submission_date"]
        ).dt.days.fillna(0).clip(lower=0)
    else:
        df["ticket_age_days"] = 0

    df.reset_index(drop=True, inplace=True)
    print(f"[DATA] Loaded {len(df):,} tickets")
    print(f"[DATA] Priority distribution:\n{df['ticket_priority'].value_counts().to_string()}")
    print(f"[DATA] Issue categories:\n{df['ticket_type'].value_counts().to_string()}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PSEUDO-LABEL GENERATION (SELF-SUPERVISED, 3 SIGNALS)
# ════════════════════════════════════════════════════════════════════════════

# ── Keyword lists tuned to your dataset's Issue_Category values
# (Technical, Billing, Account, General Inquiry)

CRITICAL_KEYWORDS = [
    "outage", "down", "crash", "crashes", "critical", "urgent", "emergency",
    "data loss", "breach", "security", "cannot access", "not working", "broken",
    "failed", "failure", "immediately", "asap", "production", "revenue loss",
    "sla breach", "escalate", "blocked", "system down", "corrupted",
    "unresponsive", "severe", "disaster", "exploit", "vulnerability",
    "hacked", "ransomware", "service unavailable", "500 error", "database error",
    "cannot log", "cannot login", "locked out", "account compromised",
    "permanently delete", "data breach", "lost all data"
]
HIGH_KEYWORDS = [
    "error", "issue", "problem", "slow", "delay", "bug", "not loading",
    "freezing", "incorrect", "wrong", "missing", "unable to", "cannot",
    "keeps failing", "intermittent", "unexpected", "degraded", "performance",
    "timeout", "high latency", "partial", "stuck", "broken feature",
    "not syncing", "sync issue", "login failed", "password reset",
    "payment failed", "charge failed", "subscription", "auto renewed",
    "2fa", "two factor", "authentication", "dashboard", "spinning wheel",
    "settings tab", "application crashes", "update failed"
]
MEDIUM_KEYWORDS = [
    "question", "inquiry", "how do i", "help", "clarification", "guide",
    "assistance", "not sure", "confused", "wondering", "would like",
    "request", "enhancement", "feature request", "improvement",
    "upgrade", "enterprise plan", "roadmap", "new features",
    "refund status", "payment method", "update payment"
]
LOW_KEYWORDS = [
    "feedback", "suggestion", "compliment", "thank", "appreciate",
    "great job", "well done", "satisfied", "happy", "love", "excellent",
    "headquarters", "office location", "hours of operation", "where is",
    "product question", "general question"
]
ESCALATION_PHRASES = [
    "escalate", "speak to manager", "supervisor", "legal action",
    "lawsuit", "unacceptable", "worst", "terrible", "horrible",
    "demand refund", "cancel subscription", "chargeback", "cancel account",
    "delete my account", "permanently delete"
]

# ── Category severity base scores (from Issue_Category column)
CATEGORY_SEVERITY = {
    "technical":       0.75,
    "billing":         0.55,
    "account":         0.50,
    "general inquiry": 0.15,
    "unknown":         0.30,
}


def compute_nlp_severity(text: str, category: str = "unknown") -> float:
    """Returns 0–1 severity score from rule-based NLP + category context."""
    text_lower = text.lower()

    # Base score from category
    cat_key = category.lower().strip() if category else "unknown"
    base = CATEGORY_SEVERITY.get(cat_key, 0.30)

    score = base

    # Keyword scoring
    crit_count = sum(1 for kw in CRITICAL_KEYWORDS if kw in text_lower)
    high_count  = sum(1 for kw in HIGH_KEYWORDS    if kw in text_lower)
    med_count   = sum(1 for kw in MEDIUM_KEYWORDS  if kw in text_lower)
    low_count   = sum(1 for kw in LOW_KEYWORDS     if kw in text_lower)

    score += crit_count * 0.30
    score += high_count  * 0.15
    score += med_count   * 0.03
    score -= low_count   * 0.12

    # Negation
    neg_count = len(re.findall(
        r"\b(not|no|never|cannot|can't|won't|didn't|doesn't|isn't|aren't|wasn't)\b",
        text_lower
    ))
    score += neg_count * 0.04

    # Escalation phrases
    esc_count = sum(1 for phrase in ESCALATION_PHRASES if phrase in text_lower)
    score += esc_count * 0.20

    # Punctuation urgency
    score += text.count("!") * 0.03
    if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.4:
        score += 0.10

    return float(np.clip(score, 0.0, 1.0))


def nlp_signal(df: pd.DataFrame) -> np.ndarray:
    """Signal 1: Rule-based NLP + category-aware severity."""
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[SIG1] NLP Signal"):
        s = compute_nlp_severity(
            str(row["full_text"]),
            str(row.get("ticket_type", "unknown"))
        )
        scores.append(s)
    return np.array(scores)


def embedding_cluster_signal(df: pd.DataFrame, n_clusters: int = 5):
    """
    Signal 2: Sentence-BERT embeddings + KMeans clustering.
    Clusters ranked by mean NLP score to assign severity.
    n_clusters=5 to capture: Critical/High/Medium/Low/General gradations.
    """
    print("[SIG2] Computing sentence embeddings (this takes a few minutes)...")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert.encode(
        df["full_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"[SIG2] KMeans clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Rank clusters by mean NLP score
    nlp_scores = np.array([
        compute_nlp_severity(t, c)
        for t, c in zip(df["full_text"], df["ticket_type"])
    ])
    cluster_mean = {
        c: nlp_scores[cluster_labels == c].mean()
        for c in range(n_clusters)
    }
    ranked = sorted(cluster_mean, key=cluster_mean.get)
    rank_map = {c: i / max(n_clusters - 1, 1) for i, c in enumerate(ranked)}
    embed_scores = np.array([rank_map[c] for c in cluster_labels])

    joblib.dump(
        {"kmeans": kmeans, "rank_map": rank_map, "sbert_name": "all-MiniLM-L6-v2"},
        f"{MODEL_DIR}/clustering.pkl"
    )
    print(f"[SIG2] Cluster severity map: { {c: f'{v:.3f}' for c,v in rank_map.items()} }")
    return embed_scores, embeddings


def resolution_time_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Signal 3: Resolution time (hours).
    Longer resolution → higher inferred severity.
    Your dataset has values like 7, 27, 40, 41, 43, 64, 65, 92, 106, 110 hours.
    """
    rt = df["resolution_time"].values.astype(float)
    rt_min, rt_max = rt.min(), rt.max()
    if rt_max == rt_min:
        return np.full(len(df), 0.5)
    return (rt - rt_min) / (rt_max - rt_min)


def satisfaction_signal(df: pd.DataFrame) -> np.ndarray:
    """
    Bonus Signal 4 (unique to your dataset): Satisfaction score.
    LOW satisfaction (1-2) → likely high severity / mismatch.
    HIGH satisfaction (4-5) → likely correctly handled.
    Inverted so high severity = high score.
    """
    sat = df["satisfaction_score"].values.astype(float)
    sat_min, sat_max = sat.min(), sat.max()
    if sat_max == sat_min:
        return np.full(len(df), 0.5)
    normalized = (sat - sat_min) / (sat_max - sat_min)
    return 1.0 - normalized  # invert: low satisfaction = high severity


# ── Priority → Numeric (your dataset: Low/Medium/High, possibly Critical)
PRIORITY_TO_NUM = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
NUM_TO_PRIORITY = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


def num_to_severity_label(score: float) -> str:
    """Convert 0-1 fused score to priority label."""
    if score >= 0.72:   return "Critical"
    elif score >= 0.50: return "High"
    elif score >= 0.28: return "Medium"
    else:               return "Low"


def generate_pseudo_labels(df: pd.DataFrame) -> tuple:
    """
    Fuse 4 signals with weighted average:
      Signal 1 — NLP + Category (35%)
      Signal 2 — Embedding Cluster (30%)
      Signal 3 — Resolution Time (20%)
      Signal 4 — Satisfaction Score (15%)  ← unique to your dataset
    """
    sig1 = nlp_signal(df)
    sig2, embeddings = embedding_cluster_signal(df)
    sig3 = resolution_time_signal(df)
    sig4 = satisfaction_signal(df)

    W1, W2, W3, W4 = 0.35, 0.30, 0.20, 0.15
    fused = W1*sig1 + W2*sig2 + W3*sig3 + W4*sig4

    df["sig1_nlp"]       = sig1
    df["sig2_embed"]     = sig2
    df["sig3_restime"]   = sig3
    df["sig4_satisf"]    = sig4
    df["fused_score"]    = fused

    df["inferred_severity"] = pd.Series(fused).apply(num_to_severity_label).values
    df["inferred_num"]      = df["inferred_severity"].map(PRIORITY_TO_NUM)
    df["assigned_num"]      = df["ticket_priority"].map(PRIORITY_TO_NUM).fillna(1)
    df["severity_delta"]    = df["inferred_num"] - df["assigned_num"]
    df["mismatch_label"]    = (df["severity_delta"].abs() >= 1).astype(int)
    df["mismatch_type"]     = df.apply(
        lambda r: "Hidden Crisis" if r["severity_delta"] > 0
        else ("False Alarm" if r["severity_delta"] < 0 else "Consistent"),
        axis=1
    )

    # ── Signal agreement
    b1 = (sig1 >= 0.5).astype(int)
    b2 = (sig2 >= 0.5).astype(int)
    b3 = (sig3 >= 0.5).astype(int)
    b4 = (sig4 >= 0.5).astype(int)
    print(f"\n[PSEUDO] Signal agreements:")
    print(f"  S1↔S2: {(b1==b2).mean():.3f} | S1↔S3: {(b1==b3).mean():.3f}")
    print(f"  S1↔S4: {(b1==b4).mean():.3f} | S2↔S3: {(b2==b3).mean():.3f}")
    print(f"\n[PSEUDO] Mismatch rate: {df['mismatch_label'].mean():.2%}")
    print(f"[PSEUDO] Label dist: {dict(df['mismatch_label'].value_counts())}")
    print(f"[PSEUDO] Mismatch types:\n{df['mismatch_type'].value_counts().to_string()}")

    df.to_csv(f"{RESULTS_DIR}/pseudo_labeled.csv", index=False)
    return df, embeddings


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Features:
      - TF-IDF on full_text (5000 features, bigrams)
      - Structured: channel, issue_category, agent, resolution_time,
                    satisfaction_score, ticket_age_days, all 4 signals
    """
    # ── TF-IDF
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )
    X_tfidf = tfidf.fit_transform(df["full_text"])

    # ── Label encoders for categorical columns
    le_channel  = LabelEncoder()
    le_type     = LabelEncoder()
    le_agent    = LabelEncoder()

    df["channel_enc"] = le_channel.fit_transform(df["ticket_channel"].astype(str))
    df["type_enc"]    = le_type.fit_transform(df["ticket_type"].astype(str))
    df["agent_enc"]   = le_agent.fit_transform(
        df.get("assigned_agent", pd.Series(["unknown"]*len(df))).astype(str)
    )

    # ── Email domain as customer-tier proxy
    df["email_domain"] = df["customer_email"].astype(str).apply(
        lambda e: e.split("@")[-1].split(".")[0] if "@" in str(e) else "unknown"
    )
    le_domain = LabelEncoder()
    df["domain_enc"] = le_domain.fit_transform(df["email_domain"].astype(str))

    # ── Structured feature columns (text + metadata, all signals + satisfaction)
    structured_cols = [
        "channel_enc", "type_enc", "agent_enc", "domain_enc",
        "resolution_time", "satisfaction_score", "ticket_age_days",
        "sig1_nlp", "sig2_embed", "sig3_restime", "sig4_satisf", "fused_score"
    ]

    scaler = StandardScaler()
    X_struct = scaler.fit_transform(df[structured_cols].fillna(0))
    X_struct_sparse = csr_matrix(X_struct)

    X = hstack([X_tfidf, X_struct_sparse])
    y = df["mismatch_label"].values

    # Save all encoders
    joblib.dump({
        "tfidf":        tfidf,
        "le_channel":   le_channel,
        "le_type":      le_type,
        "le_agent":     le_agent,
        "le_domain":    le_domain,
        "scaler":       scaler,
        "structured_cols": structured_cols
    }, f"{MODEL_DIR}/feature_encoders.pkl")

    print(f"\n[FEATURES] Matrix shape: {X.shape}")
    print(f"[FEATURES] Class distribution: {dict(Counter(y))}")
    return X, y, df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLASSIFIER TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train_classifier(X, y: np.ndarray) -> dict:
    """
    GradientBoostingClassifier on pseudo-labeled data.
    SMOTE for class imbalance. 5-fold CV + held-out eval.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # SMOTE — handle case where minority class is very small
    minority_count = min(Counter(y_train).values())
    k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

    print(f"[TRAIN] Applying SMOTE (k_neighbors={k_neighbors})...")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[TRAIN] After SMOTE: {dict(Counter(y_res))}")

    # Model — tuned for this dataset size
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42
    )

    print("[TRAIN] Running 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_res, y_res, cv=5, scoring="f1_macro")
    print(f"[TRAIN] CV Macro F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("[TRAIN] Fitting final model...")
    clf.fit(X_res, y_res)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Per-class recall — handle case where only one class exists in test
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    recall_0 = float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0
    recall_1 = float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0

    print("\n" + "="*60)
    print("  EVALUATION RESULTS (Held-out 20%)")
    print("="*60)
    print(f"  Accuracy           : {acc:.4f}  (≥0.83 required)")
    print(f"  Macro F1           : {f1:.4f}  (≥0.82 required)")
    print(f"  Recall[Consistent] : {recall_0:.4f}  (≥0.78 required)")
    print(f"  Recall[Mismatched] : {recall_1:.4f}  (≥0.78 required)")
    print("="*60)
    print(classification_report(
        y_test, y_pred,
        target_names=["Consistent", "Mismatched"],
        zero_division=0
    ))

    joblib.dump(clf, f"{MODEL_DIR}/classifier.pkl")

    metrics = {
        "accuracy":           round(acc,      4),
        "macro_f1":           round(f1,       4),
        "recall_consistent":  round(recall_0, 4),
        "recall_mismatch":    round(recall_1, 4),
        "cv_f1_mean":         round(float(cv_scores.mean()), 4),
        "cv_f1_std":          round(float(cv_scores.std()),  4),
    }
    with open(f"{RESULTS_DIR}/metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    return {
        "clf": clf, "X_test": X_test, "y_test": y_test,
        "y_pred": y_pred, "metrics": metrics
    }


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVIDENCE DOSSIER GENERATION
# ════════════════════════════════════════════════════════════════════════════

def extract_keywords_from_text(text: str) -> list:
    """Extract only keywords that actually appear in the ticket text."""
    text_lower = text.lower()
    found = []
    for kw in CRITICAL_KEYWORDS:
        if kw in text_lower:
            found.append({"keyword": kw, "tier": "critical"})
    for kw in HIGH_KEYWORDS:
        if kw in text_lower:
            found.append({"keyword": kw, "tier": "high"})
    for kw in ESCALATION_PHRASES:
        if kw in text_lower:
            found.append({"keyword": kw, "tier": "escalation"})
    return found[:5]


def generate_dossier(row: pd.Series) -> dict:
    """
    Hallucination-free Evidence Dossier.
    Every claim traceable to a specific ticket field.
    """
    assigned   = row["ticket_priority"]
    inferred   = row["inferred_severity"]
    delta      = int(row.get("severity_delta", 0))
    mtype      = row["mismatch_type"]
    confidence = float(np.clip(row["fused_score"], 0.0, 1.0))
    text       = str(row["full_text"])
    category   = str(row.get("ticket_type", "unknown"))
    rt         = float(row.get("resolution_time", 24.0))
    sat        = float(row.get("satisfaction_score", 3.0))
    channel    = str(row.get("ticket_channel", "unknown"))
    agent      = str(row.get("assigned_agent", "unknown"))

    feature_evidence = []

    # ── Evidence 1: NLP keywords (from ticket_subject + ticket_description)
    kws = extract_keywords_from_text(text)
    kw_str = ", ".join([k["keyword"] for k in kws]) if kws else "no strong keywords detected"
    feature_evidence.append({
        "signal":       "keyword_nlp",
        "value":        kw_str,
        "weight":       f"{row.get('sig1_nlp', 0):.3f}",
        "source_field": "Ticket_Subject + Ticket_Description"
    })

    # ── Evidence 2: Issue category base severity
    cat_sev = CATEGORY_SEVERITY.get(category.lower(), 0.30)
    feature_evidence.append({
        "signal":       "issue_category",
        "value":        f"{category} (base severity: {cat_sev:.2f})",
        "weight":       "0.35 (combined with NLP)",
        "source_field": "Issue_Category"
    })

    # ── Evidence 3: Embedding cluster
    feature_evidence.append({
        "signal":       "embedding_cluster",
        "value":        f"Semantic cluster severity score: {row.get('sig2_embed', 0):.3f}",
        "weight":       f"{row.get('sig2_embed', 0):.3f}",
        "source_field": "Ticket_Subject + Ticket_Description (semantic)"
    })

    # ── Evidence 4: Resolution time
    rt_interp = (
        "Very high (>72h): strong indicator of critical underlying issue"
        if rt > 72 else
        "High (48–72h): suggests complex or high-severity issue"
        if rt > 48 else
        "Moderate (24–48h): consistent with medium-high severity"
        if rt > 24 else
        "Low (<24h): suggests routine or low-severity issue"
    )
    feature_evidence.append({
        "signal":         "resolution_time",
        "value":          f"{rt:.1f} hours",
        "interpretation": rt_interp,
        "source_field":   "Resolution_Time_Hours"
    })

    # ── Evidence 5: Satisfaction score (unique to your dataset)
    sat_interp = (
        "Very low (1–2): customer highly dissatisfied, suggests mishandled ticket"
        if sat <= 2 else
        "Below average (3): moderate dissatisfaction"
        if sat <= 3 else
        "Satisfactory (4–5): customer reasonably satisfied"
    )
    feature_evidence.append({
        "signal":         "satisfaction_score",
        "value":          f"{sat:.0f}/5",
        "interpretation": sat_interp,
        "source_field":   "Satisfaction_Score"
    })

    # ── Evidence 6: Channel
    channel_weight = {
        "phone": "high urgency channel",
        "chat":  "real-time channel, moderate urgency",
        "email": "async channel, lower urgency signal",
        "web form": "async channel, lower urgency signal",
        "social media": "public channel, reputation risk"
    }
    ch_interp = channel_weight.get(channel.lower(), "standard channel")
    feature_evidence.append({
        "signal":       "channel",
        "value":        f"{channel} — {ch_interp}",
        "weight":       "0.05",
        "source_field": "Ticket_Channel"
    })

    # ── Constraint analysis — fully grounded on ticket data
    top_kw_str = ", ".join([k["keyword"] for k in kws[:2]]) if kws else "none detected"

    if mtype == "Hidden Crisis":
        constraint = (
            f"Ticket '{row.get('ticket_id', 'N/A')}' (category: {category}) was assigned "
            f"'{assigned}' priority, but multi-signal analysis infers '{inferred}' severity. "
            f"Key indicators: [{top_kw_str}] detected in description; "
            f"resolution required {rt:.0f} hours; satisfaction score was {sat:.0f}/5. "
            f"This under-prioritization risks SLA breach and customer churn."
        )
    elif mtype == "False Alarm":
        constraint = (
            f"Ticket '{row.get('ticket_id', 'N/A')}' (category: {category}) was assigned "
            f"'{assigned}' priority, but analysis infers only '{inferred}' severity. "
            f"Text signals are weak (keywords: [{top_kw_str}]); "
            f"resolution took {rt:.0f} hours with satisfaction {sat:.0f}/5. "
            f"Over-escalation wastes high-priority agent capacity."
        )
    else:
        constraint = (
            f"Assigned priority '{assigned}' aligns with inferred severity '{inferred}'. "
            f"No significant mismatch detected for ticket {row.get('ticket_id', 'N/A')}."
        )

    return {
        "ticket_id":          str(row.get("ticket_id", "N/A")),
        "assigned_priority":  assigned,
        "inferred_severity":  inferred,
        "mismatch_type":      mtype,
        "severity_delta":     f"{'+' if delta > 0 else ''}{delta}",
        "feature_evidence":   feature_evidence,
        "constraint_analysis": constraint,
        "confidence":         f"{confidence:.4f}",
        "channel":            channel,
        "issue_category":     category,
        "assigned_agent":     agent,
        "resolution_hours":   rt,
        "satisfaction_score": sat
    }


def generate_all_dossiers(df: pd.DataFrame) -> list:
    flagged  = df[df["mismatch_label"] == 1].copy()
    dossiers = []
    for _, row in tqdm(flagged.iterrows(), total=len(flagged),
                       desc="[DOSSIER] Generating"):
        dossiers.append(generate_dossier(row))

    with open(f"{RESULTS_DIR}/dossiers.json", "w") as fh:
        json.dump(dossiers, fh, indent=2)

    print(f"[DOSSIER] Generated {len(dossiers)} dossiers → {RESULTS_DIR}/dossiers.json")
    return dossiers


# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ABLATION STUDY
# ════════════════════════════════════════════════════════════════════════════

def run_ablation(df: pd.DataFrame, y: np.ndarray) -> dict:
    print("\n[ABLATION] Running signal ablation study...")

    signal_sets = {
        "NLP Only":              ["sig1_nlp"],
        "Embedding Only":        ["sig2_embed"],
        "ResTime Only":          ["sig3_restime"],
        "Satisfaction Only":     ["sig4_satisf"],
        "NLP + Embed":           ["sig1_nlp", "sig2_embed"],
        "NLP + ResTime":         ["sig1_nlp", "sig3_restime"],
        "NLP + Satisfaction":    ["sig1_nlp", "sig4_satisf"],
        "NLP + Embed + ResTime": ["sig1_nlp", "sig2_embed", "sig3_restime"],
        "All 4 Signals":         ["sig1_nlp", "sig2_embed", "sig3_restime", "sig4_satisf"],
    }

    results = {}
    for name, cols in signal_sets.items():
        fused      = df[cols].mean(axis=1).values
        y_ablation = (fused >= 0.5).astype(int)
        acc = accuracy_score(y, y_ablation)
        f1  = f1_score(y, y_ablation, average="macro", zero_division=0)
        results[name] = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}
        print(f"  {name:<28}: Acc={acc:.4f}  MacroF1={f1:.4f}")

    with open(f"{RESULTS_DIR}/ablation.json", "w") as fh:
        json.dump(results, fh, indent=2)
    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  SUPPORT INTEGRITY AUDITOR — TRAINING PIPELINE")
    print("  Dataset: enhanced_customer_support_data.csv / tickets.csv")
    print("="*60 + "\n")

    df                   = load_and_preprocess()
    df, embeddings       = generate_pseudo_labels(df)
    X, y, df             = engineer_features(df)
    results              = train_classifier(X, y)
    dossiers             = generate_all_dossiers(df)
    ablation             = run_ablation(df, y)

    print("\n[DONE] Pipeline complete.")
    print(f"  Models  → {MODEL_DIR}")
    print(f"  Results → {RESULTS_DIR}")
    return df, results, dossiers


if __name__ == "__main__":
    main()