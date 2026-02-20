import calendar
import datetime as dt
import math
import random
import re
from typing import Dict, List, Optional, Tuple

import gspread
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Workout Dashboard", layout="wide")

METRICS = [
    "Plank_min",
    "Squats",
    "Crunches",
    "Pushups",
    "Pullups",
    "Skips_min",
    "Stairs",
    "Running_km",
    "Cult_sessions",
]
SHEET_COLS = ["Date", *METRICS, "Total_score"]
CFG_COLS = ["Metric", "Weight"]
REQ_WEIGHTS = {
    "Plank_min": 30.0,
    "Squats": 1.0,
    "Crunches": 1.0,
    "Pushups": 1.0,
    "Pullups": 2.0,
    "Skips_min": 20.0,
    "Stairs": 7.0,
    "Running_km": 100.0,
    "Cult_sessions": 800.0,
}
UNITS = {
    "Plank_min": "min",
    "Squats": "reps",
    "Crunches": "reps",
    "Pushups": "reps",
    "Pullups": "reps",
    "Skips_min": "min",
    "Stairs": "floors",
    "Running_km": "km",
    "Cult_sessions": "sessions",
}
CAT = {
    "Plank_min": "Strength",
    "Squats": "Strength",
    "Crunches": "Strength",
    "Pushups": "Strength",
    "Pullups": "Strength",
    "Stairs": "Strength",
    "Skips_min": "Running/Cardio",
    "Running_km": "Running/Cardio",
    "Cult_sessions": "Cult",
}
COL = {
    "line": "rgba(125,150,180,.26)",
    "text": "#e8eef8",
    "muted": "#9db0cc",
    "total": "#5CFF9D",
    "strength": "#58A8FF",
    "cardio": "#FFA452",
    "cult": "#BE8CFF",
    "danger": "#FF6D7B",
    "warn": "#FFC75A",
}
CHART_CFG = {"displayModeBar": False}
QUOTES = [
    "Consistency compounds faster than intensity.",
    "Do clean days. Compounding handles the rest.",
    "Momentum is built one repeatable day at a time.",
    "Play the long game; keep the streak alive.",
]


def theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
        .stApp {{
            font-family: "Outfit", sans-serif; color:{COL["text"]};
            background: radial-gradient(900px 420px at 92% -6%, rgba(92,255,157,.12), transparent 58%),
                        radial-gradient(800px 400px at -8% 3%, rgba(88,168,255,.10), transparent 58%),
                        linear-gradient(145deg,#060915,#0b1327);
        }}
        .stMarkdown p, .stMarkdown li, label, .stMetricLabel, .stMetricValue {{color:{COL['text']} !important;}}
        .block-container {{padding-top:.55rem; padding-bottom:.75rem; max-width:1520px;}}
        section[data-testid="stSidebar"] > div {{background:linear-gradient(180deg,#0f172c,#121d36); border-right:1px solid rgba(125,150,180,.38);}}
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {{color:#e9f1ff !important;}}
        [data-testid="stHeader"] {{background:rgba(0,0,0,0);}}
        div[data-testid="stForm"] {{background:rgba(10,14,28,.92); border:1px solid {COL["line"]}; border-radius:14px; padding:.8rem .8rem .3rem .8rem;}}
        div[data-testid="stVerticalBlockBorderWrapper"] {{background:linear-gradient(180deg,rgba(15,21,40,.88),rgba(9,13,28,.92)); border:1px solid {COL["line"]} !important; border-radius:14px; padding:.5rem .62rem; box-shadow:0 8px 20px rgba(0,0,0,.25);}}
        div[data-baseweb="input"] > div {{background:rgba(35,45,72,.95) !important; border:1px solid rgba(190,205,236,.58) !important; border-radius:10px !important;}}
        div[data-baseweb="input"] input {{color:{COL["text"]} !important;}}
        .stDateInput input {{color:{COL["text"]} !important;}}
        .stButton > button {{background:rgba(18,26,48,.95); color:{COL["text"]}; border:1px solid rgba(125,150,180,.4); border-radius:10px; font-weight:600;}}
        .stButton > button:hover {{border-color:rgba(92,255,157,.65); color:#fff;}}
        .stFormSubmitButton > button {{background:#5CFF9D !important; color:#041223 !important; border:1px solid #7dffb3 !important; font-weight:800 !important;}}
        .stFormSubmitButton > button:hover {{background:#7dffb3 !important; color:#03101f !important;}}
        div[data-testid="stMetric"] {{background:rgba(14,20,38,.6); border:1px solid {COL["line"]}; border-radius:12px;}}
        .kpi {{border:1px solid {COL["line"]}; border-radius:14px; padding:.56rem .66rem; min-height:132px; height:132px; display:flex; flex-direction:column; justify-content:space-between;}}
        .k1 {{background:linear-gradient(145deg,rgba(32,71,52,.56),rgba(12,27,24,.72));}}
        .k2 {{background:linear-gradient(145deg,rgba(24,69,86,.56),rgba(12,25,35,.74));}}
        .k3 {{background:linear-gradient(145deg,rgba(88,62,20,.58),rgba(30,20,12,.74));}}
        .k4p {{background:linear-gradient(145deg,rgba(25,72,55,.58),rgba(12,25,20,.72));}}
        .k4n {{background:linear-gradient(145deg,rgba(84,35,43,.58),rgba(31,13,18,.72));}}
        .k4z {{background:linear-gradient(145deg,rgba(57,63,74,.55),rgba(22,27,36,.72));}}
        .k5 {{background:linear-gradient(145deg,rgba(56,37,86,.60),rgba(20,13,32,.76));}}
        .kh {{font-size:.9rem; letter-spacing:.01em; text-transform:none; font-weight:700; margin:0;}}
        .kv {{font-size:2.28rem; font-weight:800; margin:.16rem 0 .04rem 0; line-height:1;}}
        .ks {{font-size:.8rem; color:{COL["muted"]}; margin:0;}}
        .sec-k {{color:#7cf6bb; letter-spacing:.12em; text-transform:uppercase; font-size:.68rem; font-weight:700; margin:0 0 .12rem 0;}}
        .sec-t {{font-size:1.18rem; font-weight:700; margin:0;}}
        .sec-s {{font-size:.79rem; color:{COL["muted"]}; margin:.1rem 0 0 0;}}
        .pt {{font-size:.92rem; font-weight:700; margin:0;}}
        .pn {{font-size:.79rem; color:{COL["muted"]}; margin:.05rem 0 0 0;}}
        .qotd {{font-size:.86rem; color:#c7d7f5; margin:.12rem 0 0 0; font-style:italic;}}
        .wrow {{display:flex; justify-content:space-between; border-bottom:1px solid rgba(125,150,180,.15); padding:.26rem 0; font-size:.88rem;}}
        .pill {{display:inline-block; border:1px solid {COL["line"]}; border-radius:999px; padding:.12rem .55rem; margin:.12rem .2rem 0 0; font-size:.74rem; background:rgba(20,29,55,.72);}}
        .insight {{padding:.7rem .8rem; border:1px solid {COL['line']}; border-radius:12px; background:rgba(12,20,39,.72); margin:.28rem 0;}}
        .insight strong {{color:#fff;}}
        .rec {{border:1px solid {COL["line"]}; border-radius:12px; padding:.45rem .52rem; margin:.2rem 0; background:rgba(13,20,37,.86);}}
        .rec .rt {{font-size:.79rem; color:{COL["muted"]}; margin:0;}}
        .rec .rv {{font-size:1.18rem; font-weight:800; margin:.05rem 0;}}
        .rec .rs {{font-size:.74rem; color:{COL["muted"]}; margin:0;}}
        .rec0 {{background:linear-gradient(145deg,rgba(22,163,74,.22),rgba(12,33,24,.86)); box-shadow: inset 0 0 0 1px rgba(34,197,94,.75);}}
        .rec1 {{background:linear-gradient(145deg,rgba(16,185,129,.2),rgba(12,31,30,.86)); box-shadow: inset 0 0 0 1px rgba(20,184,166,.75);}}
        .rec2 {{background:linear-gradient(145deg,rgba(6,182,212,.2),rgba(10,26,35,.86)); box-shadow: inset 0 0 0 1px rgba(56,189,248,.75);}}
        .rec3 {{background:linear-gradient(145deg,rgba(59,130,246,.2),rgba(10,20,36,.86)); box-shadow: inset 0 0 0 1px rgba(96,165,250,.75);}}
        .rec4 {{background:linear-gradient(145deg,rgba(245,158,11,.22),rgba(35,22,10,.86)); box-shadow: inset 0 0 0 1px rgba(251,191,36,.78);}}
        .rec5 {{background:linear-gradient(145deg,rgba(249,115,22,.24),rgba(36,20,10,.86)); box-shadow: inset 0 0 0 1px rgba(251,146,60,.8);}}
        .rec6 {{background:linear-gradient(145deg,rgba(239,68,68,.25),rgba(35,14,16,.86)); box-shadow: inset 0 0 0 1px rgba(248,113,113,.84);}}
        .recNA {{background:linear-gradient(145deg,rgba(100,116,139,.2),rgba(18,25,36,.86)); box-shadow: inset 0 0 0 1px rgba(148,163,184,.58);}}
        .ch {{border:1px solid {COL["line"]}; border-radius:12px; padding:.52rem .58rem; background:rgba(12,19,35,.84); min-height:95px;}}
        .ch .ct {{font-size:.82rem; font-weight:700; margin:0;}}
        .ch .cv {{font-size:1.2rem; font-weight:800; margin:.08rem 0;}}
        .ch .cs {{font-size:.76rem; color:{COL["muted"]}; margin:0;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def fig_style(fig: go.Figure, h: int) -> go.Figure:
    fig.update_layout(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,24,.46)",
        margin=dict(l=0, r=0, t=8, b=0),
        font=dict(family="Outfit, sans-serif", size=12, color=COL["text"]),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color=COL["muted"])),
        yaxis=dict(showgrid=True, gridcolor="rgba(125,150,180,.18)", zeroline=False, tickfont=dict(color=COL["muted"])),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right", yanchor="bottom"),
    )
    return fig


def mlabel(m: str) -> str:
    return m.replace("_min", " (min)").replace("_km", " (km)").replace("_sessions", " sessions").replace("_", " ")


def rotating_quote() -> str:
    # Rotate by minute so quote changes dynamically during the day.
    i = int(dt.datetime.now().timestamp() // 60) % len(QUOTES)
    return QUOTES[i]


def parse_sheet_date(value: object) -> Optional[dt.date]:
    s = str(value).strip()
    if not s:
        return None
    # Handle Excel serial date numbers, then fallback to pandas parser.
    if re.fullmatch(r"\d+(\.0+)?", s):
        try:
            serial = int(float(s))
            if 20000 <= serial <= 80000:
                parsed = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
                if not pd.isna(parsed):
                    d = parsed.date()
                    if 2000 <= d.year <= 2100:
                        return d
        except Exception:
            pass
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return None
    d = parsed.date()
    if not (2000 <= d.year <= 2100):
        return None
    return d


def get_sheet() -> gspread.Spreadsheet:
    if "gcp_service_account" not in st.secrets or "google_sheet_url" not in st.secrets:
        raise KeyError("Missing secrets: gcp_service_account or google_sheet_url")
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
    )
    return gspread.authorize(creds).open_by_url(st.secrets["google_sheet_url"])


def ws(spreadsheet: gspread.Spreadsheet, key: str, default_name: str) -> gspread.Worksheet:
    name = st.secrets.get(key, default_name)
    try:
        return spreadsheet.worksheet(name)
    except gspread.WorksheetNotFound as exc:
        raise KeyError(f"Worksheet `{name}` not found.") from exc


def ensure_workout_header(worksheet: gspread.Worksheet) -> None:
    header = worksheet.row_values(1)
    if not header or [h.strip() for h in header[: len(SHEET_COLS)]] != SHEET_COLS:
        worksheet.update("A1:K1", [SHEET_COLS])


def ensure_config(worksheet: gspread.Worksheet) -> Dict[str, float]:
    rewrite = False
    header = worksheet.row_values(1)
    if not header or [h.strip() for h in header[:2]] != CFG_COLS:
        rewrite = True
    vals = worksheet.get_all_values()
    if len(vals) <= 1:
        rewrite = True
    else:
        cfg = pd.DataFrame(vals[1:], columns=vals[0])
        cfg.columns = [str(c).strip() for c in cfg.columns]
        if "Metric" not in cfg.columns or "Weight" not in cfg.columns:
            rewrite = True
        else:
            cfg["Metric"] = cfg["Metric"].astype(str).str.strip()
            cfg["Weight"] = pd.to_numeric(cfg["Weight"], errors="coerce")
            cur = {
                m: float(cfg.loc[cfg["Metric"] == m, "Weight"].iloc[-1])
                for m in REQ_WEIGHTS
                if not cfg.loc[cfg["Metric"] == m, "Weight"].empty
            }
            if set(cur.keys()) != set(REQ_WEIGHTS.keys()):
                rewrite = True
            else:
                rewrite = any(abs(cur[m] - REQ_WEIGHTS[m]) > 1e-9 for m in REQ_WEIGHTS)
    if rewrite:
        rows = [[m, w] for m, w in REQ_WEIGHTS.items()]
        worksheet.clear()
        worksheet.update("A1:B1", [CFG_COLS])
        worksheet.update(f"A2:B{len(rows)+1}", rows)
    vals2 = worksheet.get_all_values()
    cfg2 = pd.DataFrame(vals2[1:], columns=vals2[0])
    cfg2.columns = [str(c).strip() for c in cfg2.columns]
    cfg2["Metric"] = cfg2["Metric"].astype(str).str.strip()
    cfg2["Weight"] = pd.to_numeric(cfg2["Weight"], errors="coerce")
    out: Dict[str, float] = {}
    for m in METRICS:
        out[m] = float(cfg2.loc[cfg2["Metric"] == m, "Weight"].iloc[-1])
    return out

def load(worksheet: gspread.Worksheet, weights: Dict[str, float]) -> pd.DataFrame:
    ensure_workout_header(worksheet)
    vals = worksheet.get_all_values()
    if len(vals) <= 1:
        return pd.DataFrame(columns=SHEET_COLS)
    df = pd.DataFrame(vals[1:], columns=vals[0])
    for c in SHEET_COLS:
        if c not in df.columns:
            df[c] = 0 if c != "Date" else ""
    df = df[SHEET_COLS].copy()
    df["Date"] = df["Date"].apply(parse_sheet_date)
    df = df.dropna(subset=["Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    for c in METRICS + ["Total_score"]:
        s = df[c].astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    # Always recompute score from raw metrics so dashboard values stay deterministic.
    df["Total_score"] = (
        sum(df[m] * float(weights[m]) for m in METRICS)
        .round(2)
        .astype(float)
    )
    # Keep a single row per day in read path as a defensive guard for legacy duplicates.
    if not df.empty:
        df["_row_order"] = list(range(len(df)))
        df = (
            df.sort_values(["Date", "Total_score", "_row_order"])
            .groupby("Date", as_index=False)
            .tail(1)
            .drop(columns=["_row_order"])
        )
    return df.sort_values("Date").reset_index(drop=True)


def demo_data(weights: Dict[str, float], end: dt.date) -> pd.DataFrame:
    rng = random.Random(42)
    start = end - dt.timedelta(days=179)
    rows: List[Dict[str, float]] = []
    for date in pd.date_range(start, end, freq="D"):
        active = rng.random() > 0.2
        row: Dict[str, float] = {"Date": date}
        for m in METRICS:
            if not active:
                row[m] = 0.0
                continue
            if m in {"Plank_min", "Skips_min"}:
                row[m] = float(round(rng.uniform(4, 28), 1))
            elif m == "Running_km":
                row[m] = float(round(max(0, rng.gauss(3.6, 1.8)), 1))
            elif m == "Cult_sessions":
                row[m] = float(1 if rng.random() > 0.85 else 0)
            elif m == "Pullups":
                row[m] = float(max(0, int(rng.gauss(6, 4))))
            else:
                row[m] = float(max(0, int(rng.gauss(40, 22))))
        row["Total_score"] = round(sum(row[m] * float(weights[m]) for m in METRICS), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def score(entry: Dict[str, float], weights: Dict[str, float]) -> float:
    return round(sum(float(entry[m]) * float(weights[m]) for m in METRICS), 2)


def upsert(worksheet: gspread.Worksheet, day: dt.date, entry: Dict[str, float], weights: Dict[str, float]) -> str:
    # Normalize submitted date to canonical ISO string.
    ensure_workout_header(worksheet)
    vals = worksheet.get_all_values()

    if len(vals) <= 1:
        cur_df = pd.DataFrame(columns=SHEET_COLS + ["_row_order"])
    else:
        cur_df = pd.DataFrame(vals[1:], columns=vals[0])
        for c in SHEET_COLS:
            if c not in cur_df.columns:
                cur_df[c] = 0 if c != "Date" else ""
        cur_df = cur_df[SHEET_COLS].copy()
        cur_df["_row_order"] = list(range(len(cur_df)))
        cur_df["Date"] = cur_df["Date"].apply(parse_sheet_date)
        cur_df = cur_df.dropna(subset=["Date"])
        for c in METRICS + ["Total_score"]:
            s = cur_df[c].astype(str).str.replace(",", "", regex=False)
            cur_df[c] = pd.to_numeric(s, errors="coerce").fillna(0.0)

    if not cur_df.empty:
        # Deduplicate by date (keep row with highest Total_score; latest row on ties).
        cur_df = cur_df.sort_values(["Date", "Total_score", "_row_order"]).groupby("Date", as_index=False).tail(1)

    submitted_date = day
    existed = bool((cur_df["Date"] == submitted_date).any()) if not cur_df.empty else False

    # Fill only forward gaps from existing max date to submitted date.
    if not cur_df.empty:
        min_date = min(cur_df["Date"])
        max_date = max(cur_df["Date"])
        if submitted_date > max_date:
            gap_days = pd.date_range(max_date + dt.timedelta(days=1), submitted_date - dt.timedelta(days=1), freq="D")
            if len(gap_days) > 0:
                gap = pd.DataFrame({"Date": [g.date() for g in gap_days]})
                for m in METRICS:
                    gap[m] = 0.0
                gap["Total_score"] = 0.0
                gap["_row_order"] = -1
                cur_df = pd.concat([cur_df, gap], ignore_index=True)
        elif submitted_date < min_date:
            # Intentionally no backfill for dates earlier than current sheet minimum.
            pass

    # Overwrite if date exists, else append submitted row.
    cur_df = cur_df[cur_df["Date"] != submitted_date].copy()
    new_row = {"Date": submitted_date, **{m: float(entry[m]) for m in METRICS}}
    new_row["Total_score"] = score(entry, weights)
    new_row["_row_order"] = 10**9
    cur_df = pd.concat([cur_df, pd.DataFrame([new_row])], ignore_index=True)

    # Recompute score for all rows from metric columns for consistency.
    cur_df["Total_score"] = (
        sum(pd.to_numeric(cur_df[m], errors="coerce").fillna(0.0) * float(weights[m]) for m in METRICS)
        .round(2)
        .astype(float)
    )

    cur_df = cur_df.sort_values(["Date", "_row_order"]).reset_index(drop=True)
    out = cur_df[SHEET_COLS].copy()
    out["Date"] = out["Date"].apply(lambda x: x.isoformat() if isinstance(x, dt.date) else str(x))

    # Rewrite full sheet to guarantee chronological order and avoid row-index drift.
    worksheet.clear()
    worksheet.update("A1:K1", [SHEET_COLS])
    if not out.empty:
        worksheet.update(f"A2:K{len(out) + 1}", out.values.tolist(), value_input_option="USER_ENTERED")

    return "updated" if existed else "inserted"


def daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Date", *METRICS, "Total_score"])
    out = (
        df.assign(DateOnly=df["Date"].dt.date)
        .groupby("DateOnly", as_index=False)[[*METRICS, "Total_score"]]
        .sum()
        .rename(columns={"DateOnly": "Date"})
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out.sort_values("Date")


def scaffold(d: pd.DataFrame, s: dt.date, e: dt.date) -> pd.DataFrame:
    idx = pd.date_range(s, e, freq="D")
    out = pd.DataFrame({"Date": idx})
    if d.empty:
        for m in METRICS:
            out[m] = 0.0
        out["Total_score"] = 0.0
        out["has_entry"] = False
        return out
    out = out.merge(d, on="Date", how="left")
    for c in METRICS + ["Total_score"]:
        out[c] = out[c].fillna(0.0)
    out["has_entry"] = out["Date"].dt.date.isin(set(d["Date"].dt.date.tolist()))
    return out


def streak(active: List[dt.date], ref: Optional[dt.date] = None) -> Tuple[int, int]:
    if not active:
        return 0, 0
    if ref is None:
        ref = dt.date.today()
    a = sorted(active)
    best = run = 1
    for i in range(1, len(a)):
        if (a[i] - a[i - 1]).days == 1:
            run += 1
        else:
            best = max(best, run)
            run = 1
    best = max(best, run)
    if (ref - a[-1]).days > 1:
        return 0, best
    cur = 1
    for i in range(len(a) - 1, 0, -1):
        if (a[i] - a[i - 1]).days == 1:
            cur += 1
        else:
            break
    return cur, best


def streak_stats_continuous(df: pd.DataFrame, today: dt.date) -> Tuple[int, int, Optional[dt.date], bool]:
    if df.empty:
        return 0, 0, None, False
    x = df.copy()
    x["DateOnly"] = pd.to_datetime(x["Date"], errors="coerce", dayfirst=True).dt.date
    x = x.dropna(subset=["DateOnly"])
    if x.empty:
        return 0, 0, None, False
    # Deduplicate by date using max score for that day.
    day_max = x.groupby("DateOnly", as_index=False)["Total_score"].max()
    start = min(day_max["DateOnly"])
    days = [d.date() for d in pd.date_range(start, today, freq="D")]
    score_by_day = {r["DateOnly"]: float(r["Total_score"]) for _, r in day_max.iterrows()}
    active = [score_by_day.get(d, 0.0) > 0 for d in days]
    if not active:
        return 0, 0, None, False
    best = 0
    run = 0
    best_end: Optional[dt.date] = None
    for i, is_active in enumerate(active):
        if is_active:
            run += 1
            if run >= best:
                best = run
                best_end = days[i]
        else:
            run = 0
    # Sanity check example: trailing scores [10, 5, 0] => current_streak must be 0.
    cur = 0
    if active[-1]:
        for is_active in reversed(active):
            if is_active:
                cur += 1
            else:
                break
    return cur, best, best_end, active[-1]


def continuous_total_scores(df: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    if end < start:
        return pd.DataFrame({"Date": pd.to_datetime([]), "Total_score": pd.Series(dtype=float)})
    if df.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"Date": idx, "Total_score": [0.0] * len(idx)})
    x = df.copy()
    x["DateOnly"] = pd.to_datetime(x["Date"], errors="coerce", dayfirst=True).dt.date
    x["Total_score"] = pd.to_numeric(x["Total_score"], errors="coerce").fillna(0.0)
    x = x.dropna(subset=["DateOnly"])
    if x.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"Date": idx, "Total_score": [0.0] * len(idx)})
    # Deduplicate by day using maximum score; missing days remain zero in the continuous index.
    day_max = x.groupby("DateOnly", as_index=False)["Total_score"].max()
    idx = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"Date": idx})
    out["DateOnly"] = out["Date"].dt.date
    out = out.merge(day_max, on="DateOnly", how="left")
    out["Total_score"] = out["Total_score"].fillna(0.0).astype(float)
    return out[["Date", "Total_score"]]


def month_tbl(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["Month", "MonthStart", "Total_score", "Active"])
    m = d.copy()
    m["MonthStart"] = m["Date"].dt.to_period("M").dt.to_timestamp()
    m["Month"] = m["MonthStart"].dt.strftime("%Y-%m")
    total = m.groupby(["Month", "MonthStart"], as_index=False)["Total_score"].sum()
    active = m.loc[m["Total_score"] > 0].groupby("Month", as_index=False)["Date"].nunique().rename(columns={"Date": "Active"})
    out = total.merge(active, on="Month", how="left").fillna({"Active": 0}).sort_values("MonthStart")
    out["Active"] = out["Active"].astype(int)
    return out


def month_avg_and_rank(d: pd.DataFrame, today: dt.date) -> Tuple[float, Optional[int]]:
    if d.empty:
        return 0.0, None
    cur_start = today.replace(day=1)
    days_elapsed = (today - cur_start).days + 1
    cur_mtd = continuous_total_scores(d, cur_start, today)
    month_total = float(cur_mtd["Total_score"].sum())
    month_avg = month_total / days_elapsed if days_elapsed > 0 else 0.0

    dtmp = d.copy()
    dtmp["DateOnly"] = pd.to_datetime(dtmp["Date"], errors="coerce", dayfirst=True).dt.date
    dtmp = dtmp.dropna(subset=["DateOnly"])
    if dtmp.empty:
        return month_avg, 1
    first_day = min(dtmp["DateOnly"])
    m_start = first_day.replace(day=1)
    rows: List[Dict[str, object]] = []
    while m_start < cur_start:
        m_end = dt.date(m_start.year, m_start.month, calendar.monthrange(m_start.year, m_start.month)[1])
        m_series = continuous_total_scores(d, m_start, m_end)
        days_in_month = (m_end - m_start).days + 1
        avg_full = float(m_series["Total_score"].sum()) / days_in_month if days_in_month > 0 else 0.0
        rows.append({"MonthStart": m_start, "AvgDay": avg_full})
        m_start = (m_start.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    rows.append({"MonthStart": cur_start, "AvgDay": month_avg})

    rank = None
    if rows:
        tbl = pd.DataFrame(rows).sort_values(["AvgDay", "MonthStart"], ascending=[False, False]).reset_index(drop=True)
        cur_rows = tbl.index[tbl["MonthStart"] == cur_start]
        if len(cur_rows) > 0:
            rank = int(cur_rows[0]) + 1
    return month_avg, rank


def momentum(d: pd.DataFrame, today: dt.date) -> Tuple[Optional[float], float, float, float]:
    x = continuous_total_scores(d, today - dt.timedelta(days=27), today)
    b = x.loc[x["Date"].dt.date <= today - dt.timedelta(days=14), "Total_score"]
    a = x.loc[x["Date"].dt.date >= today - dt.timedelta(days=13), "Total_score"]
    avg_b = float(b.mean()) if not b.empty else 0.0
    avg_a = float(a.mean()) if not a.empty else 0.0
    delta = avg_a - avg_b
    if avg_b <= 0:
        return None, avg_a, avg_b, delta
    return (delta / avg_b), avg_a, avg_b, delta


def tier_from_avg56(d: pd.DataFrame, today: dt.date) -> Tuple[str, float, Optional[float]]:
    x = continuous_total_scores(d, today - dt.timedelta(days=55), today)
    avg56 = float(x["Total_score"].mean()) if not x.empty else 0.0
    if avg56 < 340:
        return "Bronze", avg56, 340.0
    if avg56 < 440:
        return "Silver", avg56, 440.0
    if avg56 < 540:
        return "Gold", avg56, 540.0
    if avg56 < 635:
        return "Platinum", avg56, 635.0
    return "Diamond", avg56, None


def last30_performance_chart(d: pd.DataFrame, today: dt.date) -> go.Figure:
    x = scaffold(d, today - dt.timedelta(days=29), today)
    x["Roll7"] = x["Total_score"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x["Date"],
            y=x["Total_score"],
            name="Daily",
            marker_color="rgba(88,168,255,.35)",
            hovertemplate="%{x|%b %d, %Y}<br>Daily: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x["Date"],
            y=x["Roll7"],
            mode="lines",
            name="7-day avg",
            line=dict(color=COL["total"], width=3),
            hovertemplate="%{x|%b %d, %Y}<br>7-day avg: %{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(dtick=7 * 24 * 60 * 60 * 1000, tickformat="%d %b")
    return fig_style(fig, 205)


def monthly_chart(m: pd.DataFrame, today: dt.date) -> go.Figure:
    if m.empty:
        return go.Figure()
    x = m.sort_values("MonthStart").copy()
    x = x.tail(12).copy()
    x["Roll3"] = x["Total_score"].rolling(3, min_periods=1).mean()
    x["Label"] = x["MonthStart"].dt.strftime("%b %Y")
    cur_m = today.strftime("%Y-%m")
    x["is_current"] = x["Month"] == cur_m
    colors = ["rgba(92,255,157,.85)" if is_cur else "rgba(42,79,144,.75)" for is_cur in x["is_current"]]
    border = [COL["total"] if is_cur else "rgba(42,79,144,.25)" for is_cur in x["is_current"]]
    x["HoverTag"] = x["is_current"].map({True: "MTD", False: "Total"})
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x["Label"],
            y=x["Total_score"],
            name="Monthly total",
            marker=dict(color=colors, line=dict(color=border, width=1.2)),
            customdata=x[["HoverTag"]],
            hovertemplate="%{x}<br>%{customdata[0]}: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(go.Scatter(x=x["Label"], y=x["Roll3"], mode="lines+markers", name="3M avg", line=dict(color=COL["cardio"], width=1.9), marker=dict(size=5)))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category")
    return fig_style(fig, 232)


def heatmap(sd: pd.DataFrame, start: dt.date, w: Dict[str, float]) -> Tuple[go.Figure, Dict[str, float], bool]:
    if sd.empty:
        return go.Figure(), {}, False
    a = start - dt.timedelta(days=start.weekday())
    x = sd.copy()
    x["week"] = x["Date"].dt.date.apply(lambda d: (d - a).days // 7)
    x["wd"] = x["Date"].dt.weekday
    nz = x.loc[x["Total_score"] > 0, "Total_score"].astype(float)
    cutoffs: Dict[str, float] = {"t1": 400.0, "t2": 600.0, "t3": 800.0, "t4": 1000.0, "t5": 1200.0, "t6": 1400.0}
    has_active = not nz.empty

    def bucket(score: float) -> int:
        if score <= 0:
            return 0
        if score <= cutoffs["t1"]:
            return 1
        if score <= cutoffs["t2"]:
            return 2
        if score <= cutoffs["t3"]:
            return 3
        if score <= cutoffs["t4"]:
            return 4
        if score <= cutoffs["t5"]:
            return 5
        if score <= cutoffs["t6"]:
            return 6
        return 7

    def intensity_label(b: int) -> str:
        if b == 0:
            return "Rest"
        if b <= 2:
            return "Low"
        if b <= 5:
            return "Medium"
        return "High"

    def top_metric_for_row(r: pd.Series) -> str:
        if float(r["Total_score"]) <= 0:
            return "No activity"
        contrib = {m: float(r[m]) * float(w[m]) for m in METRICS}
        best = max(contrib, key=contrib.get)
        return f"{mlabel(best)} ({contrib[best]:.0f} pts)"

    x["bucket"] = x["Total_score"].apply(lambda v: bucket(float(v)))
    x["intensity"] = x["bucket"].apply(lambda b: intensity_label(int(b)))
    x["top_metric"] = x.apply(top_metric_for_row, axis=1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x["week"],
            y=x["wd"],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=14,
                symbol="square",
                color=x["bucket"],
                colorscale=[
                    [0.00, "#1f2430"],
                    [1/7, "#7f1d1d"],
                    [2/7, "#ef4444"],
                    [3/7, "#f59e0b"],
                    [4/7, "#84cc16"],
                    [5/7, "#22c55e"],
                    [6/7, "#16a34a"],
                    [1.00, "#166534"],
                ],
                cmin=0,
                cmax=7,
                showscale=False,
            ),
            text=x.apply(
                lambda r: (
                    f"Date: {r['Date'].date().strftime('%b %d, %Y')}"
                    f"<br>Total score: {r['Total_score']:.1f}"
                    f"<br>Intensity: {r['intensity']}"
                    f"<br>Top metric: {r['top_metric']}"
                ),
                axis=1,
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(tickmode="array", tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], autorange="reversed")
    fig.update_xaxes(showticklabels=False)
    return fig_style(fig, 232), cutoffs, has_active


def consistency_insights(sd: pd.DataFrame) -> List[str]:
    if sd.empty:
        return [
            "No active days in last 84 days. Start with one short session today.",
            "Protect Monday: most rest days. Pre-commit a 10-minute minimum.",
            "Volatility: not enough variation to assess.",
        ]
    x = sd.copy()
    if bool((x["Total_score"] > 0).sum()) is False:
        return [
            "No active days in last 84 days. Start with one short session today.",
            "Protect Monday: most rest days. Pre-commit a 10-minute minimum.",
            "Volatility: not enough variation to assess.",
        ]
    x["Weekday"] = x["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    weekday_avg = x.groupby("Weekday", as_index=False)["Total_score"].mean()
    weekday_avg["Weekday"] = pd.Categorical(weekday_avg["Weekday"], categories=order, ordered=True)
    weekday_avg = weekday_avg.sort_values("Weekday")
    anchor = weekday_avg.loc[weekday_avg["Total_score"].idxmax(), "Weekday"]

    weekday_zero = x.groupby("Weekday", as_index=False).agg(zero_rate=("Total_score", lambda s: float((s <= 0).mean())))
    weekday_zero["Weekday"] = pd.Categorical(weekday_zero["Weekday"], categories=order, ordered=True)
    weekday_zero = weekday_zero.sort_values("Weekday")
    break_day = weekday_zero.loc[weekday_zero["zero_rate"].idxmax(), "Weekday"]

    last14 = x.tail(14)["Total_score"]
    prior28 = x.iloc[max(len(x) - 42, 0):max(len(x) - 14, 0)]["Total_score"]
    s14 = float(last14.std(ddof=0)) if not last14.empty else 0.0
    s28 = float(prior28.std(ddof=0)) if not prior28.empty else 0.0
    ratio: Optional[float] = None if s28 <= 1e-9 else (s14 / s28)

    if ratio is None:
        vtxt = "Volatility: not enough variation to assess."
    elif ratio > 1.2:
        vtxt = "Volatility up: last 2 weeks are erratic. Simplify plan for 7 days."
    elif ratio < 0.8:
        vtxt = "Volatility down: more stable. Increase difficulty on anchor day."
    else:
        vtxt = "Volatility normal: keep the same weekly structure."

    return [
        f"Anchor {anchor}: your strongest day. Put your hardest session here.",
        f"Protect {break_day}: most rest days. Pre-commit a 10-minute minimum.",
        vtxt,
    ]


def monthly_challenges(dm: pd.DataFrame, today: dt.date) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], int]:
    active_days = int((dm["Total_score"] > 0).sum()) if not dm.empty else 0
    sums = {m: float(dm[m].sum()) if (not dm.empty and m in dm.columns) else 0.0 for m in METRICS}
    total_score = float(dm["Total_score"].sum()) if not dm.empty else 0.0
    cur_streak_month = streak(sorted({x.date() for x in dm.loc[dm["Total_score"] > 0, "Date"]}), today)[0] if not dm.empty else 0

    defs = [
        {"id": "run20", "title": "Run 20 km this month", "value": sums["Running_km"], "target": 20.0, "unit": "km", "fmt": "float"},
        {"id": "push200", "title": "Do 200 pushups this month", "value": sums["Pushups"], "target": 200.0, "unit": "reps", "fmt": "int"},
        {"id": "active10", "title": "Hit 10 active days this month", "value": float(active_days), "target": 10.0, "unit": "days", "fmt": "int"},
        {"id": "stairs250", "title": "Climb 250 stairs this month", "value": sums["Stairs"], "target": 250.0, "unit": "floors", "fmt": "int"},
        {"id": "plank150", "title": "Accumulate 150 plank minutes", "value": sums["Plank_min"], "target": 150.0, "unit": "min", "fmt": "int"},
        {"id": "streak7", "title": "Build a 7-day streak", "value": float(cur_streak_month), "target": 7.0, "unit": "days", "fmt": "int"},
        {"id": "cult3", "title": "Complete 3 cult sessions", "value": sums["Cult_sessions"], "target": 3.0, "unit": "sessions", "fmt": "int"},
        {"id": "score12k", "title": "Cross 12,000 month score", "value": total_score, "target": 12000.0, "unit": "pts", "fmt": "int"},
        {"id": "pull60", "title": "Do 60 pullups this month", "value": sums["Pullups"], "target": 60.0, "unit": "reps", "fmt": "int"},
    ]
    for d in defs:
        d["done"] = float(d["value"]) >= float(d["target"])
        d["pct"] = min(float(d["value"]) / float(d["target"]), 1.0) if float(d["target"]) > 0 else 0.0
    done = [d for d in defs if bool(d["done"])]
    open_items = [d for d in defs if not bool(d["done"])]
    active = open_items[:3]
    if len(active) < 3:
        active += done[-(3 - len(active)):]
    return active, done[-3:], len(done)


def records_table(d: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        mx = float(d[m].max()) if (not d.empty and m in d.columns) else 0.0
        if mx <= 0:
            rows.append({"Metric": m, "Label": mlabel(m), "Value": 0.0, "Date": None, "Age": None, "Class": "recNA"})
            continue
        hits = d.loc[d[m] == mx].copy().sort_values("Date")
        rec_date = hits["Date"].iloc[-1].date()
        age = (today - rec_date).days
        rows.append({"Metric": m, "Label": mlabel(m), "Value": mx, "Date": rec_date, "Age": age, "Class": ""})
    out = pd.DataFrame(rows)
    valid = out.loc[out["Age"].notna(), "Age"].astype(float)
    if valid.empty:
        out.loc[out["Class"] == "", "Class"] = "recNA"
        return out
    amin, amax = float(valid.min()), float(valid.max())
    for i, r in out.iterrows():
        if pd.isna(r["Age"]):
            out.at[i, "Class"] = "recNA"
            continue
        if abs(amax - amin) < 1e-9:
            bucket = 0
        else:
            t = (float(r["Age"]) - amin) / (amax - amin)
            bucket = int(round(t * 6))
        out.at[i, "Class"] = f"rec{max(0, min(6, bucket))}"
    return out


def section(k: str, t: str, s: str) -> None:
    st.markdown(f"<p class='sec-k'>{k}</p><p class='sec-t'>{t}</p><p class='sec-s'>{s}</p>", unsafe_allow_html=True)


def ptitle(t: str, n: str = "") -> None:
    nhtml = f"<p class='pn'>{n}</p>" if n else ""
    st.markdown(f"<p class='pt'>{t}</p>{nhtml}", unsafe_allow_html=True)


theme()
weights = REQ_WEIGHTS.copy()
workout_ws: Optional[gspread.Worksheet] = None
setup_error = ""
try:
    ss = get_sheet()
    workout_ws = ws(ss, "google_worksheet_name", "Workout_DB")
    config_ws = ws(ss, "config_worksheet_name", "Config")
    weights = ensure_config(config_ws)
except Exception as ex:
    setup_error = f"{type(ex).__name__}: {ex}"

with st.sidebar:
    st.markdown("### Log Workout")
    with st.form("log_form", clear_on_submit=False):
        day = st.date_input("Date", value=dt.date.today(), format="YYYY-MM-DD")
        c1, c2 = st.columns(2)
        vals = {
            "Plank_min": c1.number_input("Plank_min", min_value=0.0, step=0.5, value=0.0),
            "Squats": c2.number_input("Squats", min_value=0.0, step=1.0, value=0.0),
        }
        c3, c4 = st.columns(2)
        vals["Crunches"] = c3.number_input("Crunches", min_value=0.0, step=1.0, value=0.0)
        vals["Pushups"] = c4.number_input("Pushups", min_value=0.0, step=1.0, value=0.0)
        c5, c6 = st.columns(2)
        vals["Pullups"] = c5.number_input("Pullups", min_value=0.0, step=1.0, value=0.0)
        vals["Skips_min"] = c6.number_input("Skips_min", min_value=0.0, step=0.5, value=0.0)
        c7, c8 = st.columns(2)
        vals["Stairs"] = c7.number_input("Stairs", min_value=0.0, step=1.0, value=0.0)
        vals["Running_km"] = c8.number_input("Running_km", min_value=0.0, step=0.1, value=0.0)
        vals["Cult_sessions"] = st.number_input("Cult_sessions", min_value=0.0, step=0.1, value=0.0)
        submit = st.form_submit_button("Submit")
    if submit:
        if workout_ws is None:
            st.warning("Read-only demo mode: connect Google Sheets to save logs.")
        else:
            mode = upsert(workout_ws, day, {k: float(v) for k, v in vals.items()}, weights)
            st.session_state["ok"] = f"Entry {mode}. Total_score recalculated from Config."
            st.rerun()
    if "ok" in st.session_state:
        st.success(st.session_state["ok"])
        del st.session_state["ok"]
    st.markdown("### Weights")
    st.markdown("".join([f"<div class='wrow'><span>{mlabel(m)}</span><strong>{weights[m]:g}</strong></div>" for m in METRICS]), unsafe_allow_html=True)
    monthly_target = st.number_input("Monthly Target", min_value=0.0, value=float(st.session_state.get("monthly_target", 25000.0)), step=500.0, key="monthly_target")
    if workout_ws is None:
        st.caption(f"Using built-in demo dataset because sheet setup failed ({setup_error}).")


df = demo_data(weights, dt.date.today()) if workout_ws is None else load(workout_ws, weights)
d_all = daily(df)
today = d_all["Date"].max().date() if not d_all.empty else dt.date.today()
min_date = today - dt.timedelta(days=89) if d_all.empty else d_all["Date"].min().date()
if "r_start" not in st.session_state or "r_end" not in st.session_state:
    st.session_state["r_start"] = max(min_date, today - dt.timedelta(days=89))
    st.session_state["r_end"] = today


def set_range(days: int) -> None:
    st.session_state["r_start"] = max(min_date, today - dt.timedelta(days=days - 1))
    st.session_state["r_end"] = today


section("Project Iron Momentum", "Execution Cockpit", "")
st.markdown(f"<p class='qotd'>\"{rotating_quote()}\"</p>", unsafe_allow_html=True)
with st.container(border=True):
    dcol, b1, b2, b3, b4 = st.columns([3.3, 1, 1, 1, 1.3], gap="small")
    with dcol:
        picked = st.date_input("Date range", value=(st.session_state["r_start"], st.session_state["r_end"]), min_value=min_date, max_value=today, format="YYYY-MM-DD", label_visibility="collapsed")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            s, e = picked
            if s > e:
                s, e = e, s
            st.session_state["r_start"], st.session_state["r_end"] = max(s, min_date), min(e, today)
    with b1:
        if st.button("Last 7", use_container_width=True):
            set_range(7); st.rerun()
    with b2:
        if st.button("30", use_container_width=True):
            set_range(30); st.rerun()
    with b3:
        if st.button("90", use_container_width=True):
            set_range(90); st.rerun()
    with b4:
        if st.button("This Month", use_container_width=True):
            st.session_state["r_start"], st.session_state["r_end"] = max(min_date, today.replace(day=1)), today; st.rerun()

if d_all.empty:
    st.info("No workout data in Workout_DB yet. Log your first day from the sidebar.")
    st.stop()

s_date, e_date = st.session_state["r_start"], st.session_state["r_end"]
hm_start = max(min_date, today - dt.timedelta(days=83))
hm_sd = scaffold(d_all, hm_start, today)
hm_insights = consistency_insights(hm_sd)
cur_streak, best_streak, best_streak_end_date, active_today = streak_stats_continuous(df, today)
m_tbl = month_tbl(d_all)
month_avg, month_rank_pos = month_avg_and_rank(df, today)
mom, mom_avg_a, mom_avg_b, _mom_delta = momentum(df, today)
tier_name, avg56, next_thr = tier_from_avg56(df, today)
streak_pct = 0 if best_streak == 0 else int(round(100.0 * cur_streak / best_streak))
best_streak_end_txt = "-" if best_streak_end_date is None else best_streak_end_date.strftime("%b %d, %Y")

k = st.columns(5, gap="small")
with k[0]:
    st.markdown(
        f"<div class='kpi k1'><p class='kh'>Day Streak</p><p class='kv'>{cur_streak}</p><p class='ks'>{'Active today' if active_today else 'Inactive today'}</p><p class='ks'>{streak_pct}% of best</p></div>",
        unsafe_allow_html=True,
    )
with k[1]:
    st.markdown(f"<div class='kpi k2'><p class='kh'>Best Streak</p><p class='kv'>{best_streak}</p><p class='ks'>Ended: {best_streak_end_txt}</p></div>", unsafe_allow_html=True)
with k[2]:
    st.markdown(
        f"<div class='kpi k3'><p class='kh'>Month Avg</p><p class='kv'>{round(month_avg):,.0f}</p><p class='ks'>Rank: {'NA' if month_rank_pos is None else '#'+str(month_rank_pos)} (avg/day)</p></div>",
        unsafe_allow_html=True,
    )
with k[3]:
    mclass = "k4z" if mom is None or abs(mom) < 1e-12 else ("k4p" if mom > 0 else "k4n")
    st.markdown(
        f"<div class='kpi {mclass}'><p class='kh'>Momentum</p><p class='kv'>{'NA' if mom is None else f'{(mom*100):+.0f}%'} </p><p class='ks'>Avg 14d: {round(mom_avg_a):,.0f}</p><p class='ks'>Prev 14d: {round(mom_avg_b):,.0f}</p></div>",
        unsafe_allow_html=True,
    )
with k[4]:
    tier_next_txt = "Top tier" if next_thr is None else f"To next: +{int(math.ceil(max(next_thr - avg56, 0.0)))} avg/day"
    st.markdown(
        f"<div class='kpi k3'><p class='kh'>Tier</p><p class='kv'>{tier_name}</p><p class='ks'>56d avg: {round(avg56):,.0f}</p><p class='ks'>{tier_next_txt}</p></div>",
        unsafe_allow_html=True,
    )

r1 = st.columns([4, 4, 4], gap="small")
with r1[0]:
    with st.container(border=True):
        s30 = scaffold(d_all, today - dt.timedelta(days=29), today)
        s30_avg = float(s30["Total_score"].mean()) if not s30.empty else 0.0
        s30_total = float(s30["Total_score"].sum()) if not s30.empty else 0.0
        s30_active = int((s30["Total_score"] > 0).sum()) if not s30.empty else 0
        ptitle("Performance")
        st.markdown(
            f"<p class='pn'>Last 30 days &bull; Avg/day: <strong>{s30_avg:,.0f}</strong> &bull; Total: <strong>{s30_total:,.0f}</strong> &bull; Active days: <strong>{s30_active}</strong></p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(last30_performance_chart(d_all, today), use_container_width=True, config=CHART_CFG)
        st.markdown("<div style='height:1px;background:rgba(125,150,180,.18);margin:.35rem 0 .28rem 0;'></div>", unsafe_allow_html=True)
        ptitle("Monthly performance", "Last 12 months")
        st.plotly_chart(monthly_chart(m_tbl, today), use_container_width=True, config=CHART_CFG)
        if not m_tbl.empty:
            b = m_tbl.loc[m_tbl["Total_score"].idxmax()]
            st.markdown(f"<p class='pn'>Best Month: {b['MonthStart'].strftime('%b %Y')} ({b['Total_score']:,.0f} pts)</p>", unsafe_allow_html=True)
with r1[1]:
    with st.container(border=True):
        ptitle("Consistency Map", "Last 84 days vs daily score bands")
        hm_fig, hm_cutoffs, hm_has_active = heatmap(hm_sd, hm_start, weights)
        st.plotly_chart(hm_fig, use_container_width=True, config=CHART_CFG)
        swatches = [
            ("Rest day", "#1f2430"),
            ("1", "#7f1d1d"),
            ("2", "#ef4444"),
            ("3", "#f59e0b"),
            ("4", "#84cc16"),
            ("5", "#22c55e"),
            ("6", "#16a34a"),
            ("7", "#166534"),
        ]
        if hm_has_active:
            legend_tips = {
                "Rest day": "Rest day: Total score is 0",
                "1": f"Very low: >0 to {hm_cutoffs['t1']:.0f}",
                "2": f"Low: {hm_cutoffs['t1']:.0f} to {hm_cutoffs['t2']:.0f}",
                "3": f"Building: {hm_cutoffs['t2']:.0f} to {hm_cutoffs['t3']:.0f}",
                "4": f"Good zone: {hm_cutoffs['t3']:.0f} to {hm_cutoffs['t4']:.0f}",
                "5": f"Strong: {hm_cutoffs['t4']:.0f} to {hm_cutoffs['t5']:.0f}",
                "6": f"Very strong: {hm_cutoffs['t5']:.0f} to {hm_cutoffs['t6']:.0f}",
                "7": f"Elite day: above {hm_cutoffs['t6']:.0f}",
            }
        else:
            legend_tips = {k: "No active days in last 84 days" for k, _ in swatches}

        legend_html = "".join(
            [
                f"<span style='display:inline-flex;align-items:center;margin:.05rem .5rem .05rem 0;font-size:.75rem;color:{COL['muted']};' title='{legend_tips[lbl]}'>"
                f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;background:{c};margin-right:.28rem;border:1px solid rgba(255,255,255,.14);'></span>{lbl}</span>"
                for lbl, c in swatches
            ]
        )
        st.markdown(f"<div>{legend_html}<span style='font-size:.75rem;color:{COL['muted']};'>Low → High</span></div>", unsafe_allow_html=True)
        for line in hm_insights[:3]:
            st.markdown(f"- {line}")
with r1[2]:
    with st.container(border=True):
        ptitle("Records / Hall of Fame")
        rec = records_table(d_all, today)
        c1, c2, c3 = st.columns(3, gap="small")
        for i, m in enumerate(METRICS):
            rr = rec.loc[rec["Metric"] == m].iloc[0]
            date_val = rr["Date"]
            age_val = rr["Age"]
            d_txt = "NA" if pd.isna(date_val) else date_val.strftime("%b %d, %Y")
            a_txt = "Age: NA" if pd.isna(age_val) else f"Age: {int(age_val)}d"
            tile = (
                f"<div class='rec {rr['Class']}'>"
                f"<p class='rt'>{rr['Label']}</p>"
                f"<p class='rv'>{float(rr['Value']):g} {UNITS[m]}</p>"
                f"<p class='rs'>Date: {d_txt}</p>"
                f"<p class='rs'>{a_txt}</p>"
                f"</div>"
            )
            [c1, c2, c3][i % 3].markdown(tile, unsafe_allow_html=True)

ms = today.replace(day=1)
me = today.replace(day=calendar.monthrange(today.year, today.month)[1])
dm = scaffold(d_all, ms, me)
mdays = (me - ms).days + 1
elapsed = (today - ms).days + 1
rem = max(mdays - elapsed, 0)
ach = float(dm["Total_score"].sum())
need = max((monthly_target - ach) / rem, 0.0) if rem > 0 and monthly_target > 0 else 0.0
proj = (ach / elapsed) * mdays if elapsed > 0 else 0.0
pace_color = COL["total"] if proj >= monthly_target else (COL["warn"] if proj >= monthly_target * 0.9 else COL["danger"])
pace_state = "On track" if proj >= monthly_target else ("Close" if proj >= monthly_target * 0.9 else "Behind")
lb14 = scaffold(d_all, today - dt.timedelta(days=13), today)
contrib = {m: float((lb14[m] * weights[m]).sum()) for m in METRICS}
weak = min(contrib, key=contrib.get)
goal_raw = max(float(lb14[weak].sum()) / 2.0 * 1.2, {"Running_km": 10, "Cult_sessions": 1}.get(weak, 20))
goal = round(goal_raw, 1) if weak in {"Running_km", "Cult_sessions"} else float(round(goal_raw))
week_start = today - dt.timedelta(days=today.weekday())
week_end = week_start + dt.timedelta(days=6)
wcur = scaffold(d_all, week_start, week_end)
prog = float(wcur[weak].sum())

last14 = scaffold(d_all, today - dt.timedelta(days=13), today)
last3 = scaffold(d_all, today - dt.timedelta(days=2), today)
avg14 = float(last14["Total_score"].mean()) if not last14.empty else 0.0
avg3 = float(last3["Total_score"].mean()) if not last3.empty else 0.0
risk = "High" if avg14 > 0 and avg3 < 0.5 * avg14 else "Low"
decision = "Push" if (mom is not None and mom > 0 and risk == "Low") else ("Recover" if (mom is not None and mom < 0 and cur_streak <= 2) else "Maintain")
active_ch, recent_done, done_count = monthly_challenges(dm, today)

coach = st.columns(3, gap="small")
with coach[0]:
    with st.container(border=True):
        ptitle("Today's Required Score")
        st.markdown(f"<p style='margin:0;font-size:2rem;font-weight:800;'>{need:,.0f}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='pn'>{pace_state} &bull; Projected {proj:,.0f}</p>", unsafe_allow_html=True)
with coach[1]:
    with st.container(border=True):
        ptitle("Break Risk")
        if avg14 <= 0:
            reason = "No baseline yet. Keep showing up for a few days to establish your rhythm."
        else:
            drop_pct = max((avg14 - avg3) / avg14 * 100.0, 0.0)
            if risk == "High":
                reason = f"Your last 3 days are about {drop_pct:.0f}% below your 14-day rhythm."
            else:
                reason = f"Your recent pace is close to your 14-day rhythm ({drop_pct:.0f}% dip)."
        st.markdown(f"<p style='margin:0;font-size:2rem;font-weight:800;color:{COL['danger'] if risk=='High' else COL['total']};'>{risk}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='pn'>{reason}</p>", unsafe_allow_html=True)
with coach[2]:
    with st.container(border=True):
        ptitle("Decision")
        st.markdown(f"<p style='margin:0;font-size:2rem;font-weight:800;color:{COL['total'] if decision=='Push' else (COL['danger'] if decision=='Recover' else COL['warn'])};'>{decision}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='pn'>Momentum {'NA' if mom is None else f'{(mom*100):+.0f}%'} &bull; Risk {risk}</p>", unsafe_allow_html=True)

with st.container(border=True):
    ptitle("Challenges", f"{done_count} closed this month")
    ch_cols = st.columns(3, gap="small")
    for i, ch in enumerate(active_ch[:3]):
        val = float(ch["value"])
        tgt = float(ch["target"])
        is_done = bool(ch["done"])
        pct = float(ch["pct"])
        fmt = str(ch["fmt"])
        val_txt = f"{val:,.1f}" if fmt == "float" else f"{int(round(val)):,.0f}"
        tgt_txt = f"{tgt:,.1f}" if fmt == "float" else f"{int(round(tgt)):,.0f}"
        with ch_cols[i]:
            st.markdown(
                f"<div class='ch'><p class='ct'>{ch['title']}</p><p class='cv'>{val_txt} / {tgt_txt} {ch['unit']}</p><p class='cs'>{'Completed ✅' if is_done else 'In progress'}</p></div>",
                unsafe_allow_html=True,
            )
            st.progress(pct, text=f"{int(round(pct * 100))}%")
    if recent_done:
        st.markdown(
            "<p class='pn'>Recently closed: "
            + " | ".join([f"✅ {d['title']}" for d in recent_done])
            + "</p>",
            unsafe_allow_html=True,
        )

