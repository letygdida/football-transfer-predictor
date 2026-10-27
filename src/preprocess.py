"""
preprocess.py
-------------
Loads raw Transfermarkt CSV files, engineers features, handles missing values,
encodes categoricals, and saves a clean modelling-ready dataset.

Key design decisions:
  - TARGET: market_value_in_eur (current value from players.csv)
  - No future data leakage: highest_market_value is dropped because it is
    only known after the current value is already set.
  - Appearance-level stats (goals, assists, etc.) are aggregated per player
    from the last 2 seasons to reflect recent form.
  - Age is derived from date_of_birth at the time of the snapshot (last_season).
  - Categorical columns (position, foot, league) are label-encoded and stored
    in a mapping file so the dashboard can reverse-translate.

Output:
  data/processed/features.csv        <- model-ready rows
  data/processed/label_encodings.json <- mapping for categorical decoding

Run:
  python src/preprocess.py
"""

import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    path = RAW / name
    if not path.exists():
        print(f"  ✗ Missing file: {path}")
        print("    Run  python src/download_data.py  first.")
        sys.exit(1)
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {name:<30}  shape={df.shape}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_age(date_of_birth: pd.Series, reference_year: pd.Series) -> pd.Series:
    """Approximate age in years at the reference season year."""
    birth_year = pd.to_datetime(date_of_birth, errors="coerce").dt.year
    return (reference_year - birth_year).astype("float32")


def aggregate_appearances(apps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-player appearance stats (last 2 seasons only to capture
    recent form, not career totals which would dominate for older players).

    Returns one row per player_id with columns:
      total_appearances, total_goals, total_assists,
      total_minutes, total_yellow_cards, total_red_cards,
      goals_per_90, assists_per_90
    """
    # Keep only the two most recent seasons per player
    apps = apps.copy()
    apps["season_year"] = pd.to_datetime(apps["date"], errors="coerce").dt.year
    recent_cutoff = apps["season_year"].max() - 1          # last 2 seasons
    apps = apps[apps["season_year"] >= recent_cutoff]

    agg = (
        apps.groupby("player_id")
        .agg(
            total_appearances=("appearance_id", "count"),
            total_goals=("goals", "sum"),
            total_assists=("assists", "sum"),
            total_minutes=("minutes_played", "sum"),
            total_yellow_cards=("yellow_cards", "sum"),
            total_red_cards=("red_cards", "sum"),
        )
        .reset_index()
    )

    # Goals and assists per 90 minutes (avoid division by zero)
    safe_min = agg["total_minutes"].clip(lower=90)
    agg["goals_per_90"]   = (agg["total_goals"]   / safe_min * 90).round(4)
    agg["assists_per_90"] = (agg["total_assists"]  / safe_min * 90).round(4)

    return agg


def encode_categorical(df: pd.DataFrame, col: str, mappings: dict) -> pd.Series:
    """
    Label-encode a categorical column.  Unknown values map to -1.
    Stores the encoding in `mappings` for later use by the dashboard.
    """
    categories = df[col].dropna().unique().tolist()
    categories.sort()
    cat_map = {cat: idx for idx, cat in enumerate(categories)}
    mappings[col] = cat_map                                  # save for JSON
    return df[col].map(cat_map).fillna(-1).astype("int32")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Transfermarkt Feature Engineering Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading raw CSVs ...")
    players     = load_csv("players.csv")
    appearances = load_csv("appearances.csv")
    clubs       = load_csv("clubs.csv")
    competitions = load_csv("competitions.csv")

    # ------------------------------------------------------------------
    # 2. Filter players — keep only those with a non-null market value
    # ------------------------------------------------------------------
    print("\n[2/5] Filtering and cleaning players ...")
    players = players[players["market_value_in_eur"].notna()].copy()
    players = players[players["market_value_in_eur"] > 0].copy()
    print(f"  Players with valid market value: {len(players):,}")

    # Drop columns that cause target leakage or are non-predictive identifiers
    LEAKAGE_COLS = [
        "highest_market_value_in_eur",  # derived from future knowledge
        "player_code",
        "image_url",
        "url",
        "agent_name",
        "first_name",
        "last_name",
        "city_of_birth",
    ]
    players.drop(columns=[c for c in LEAKAGE_COLS if c in players.columns], inplace=True)

    # ------------------------------------------------------------------
    # 3. Engineer age
    # ------------------------------------------------------------------
    ref_year = players["last_season"].fillna(datetime.now().year).astype(int)
    players["age"] = compute_age(players["date_of_birth"], ref_year)

    # Drop the raw date columns — age is the engineered feature
    players.drop(columns=["date_of_birth", "contract_expiration_date",
                           "last_season"], errors="ignore", inplace=True)

    # Filter to plausible ages (15–45)
    players = players[players["age"].between(15, 45)]
    print(f"  Players after age filter (15–45):  {len(players):,}")

    # ------------------------------------------------------------------
    # 4. Merge club / competition metadata
    # ------------------------------------------------------------------
    print("\n[3/5] Merging club and competition data ...")

    # Attach domestic competition name from clubs
    # average_age may not exist in all dataset versions — include if present
    club_base_cols = ["club_id", "domestic_competition_id", "squad_size",
                      "national_team_players"]
    if "average_age" in clubs.columns:
        club_base_cols.append("average_age")
    club_cols = clubs[club_base_cols].copy()
    players = players.merge(
        club_cols, left_on="current_club_id", right_on="club_id", how="left"
    )
    players.drop(columns=["club_id"], errors="ignore", inplace=True)

    # Resolve the competition ID to a competition type (league vs. cup etc.)
    comp_type = competitions[["competition_id", "type"]].drop_duplicates()
    players = players.merge(
        comp_type,
        left_on="current_club_domestic_competition_id",
        right_on="competition_id",
        how="left"
    )
    players.rename(columns={"type": "competition_type"}, inplace=True)
    players.drop(columns=["competition_id",
                           "current_club_domestic_competition_id",
                           "domestic_competition_id"],
                 errors="ignore", inplace=True)

    # ------------------------------------------------------------------
    # 5. Merge appearance stats
    # ------------------------------------------------------------------
    print("\n[4/5] Aggregating appearance statistics ...")
    app_stats = aggregate_appearances(appearances)
    players = players.merge(app_stats, on="player_id", how="left")

    # Players with no recorded appearances get zeroed stats
    stat_cols = ["total_appearances", "total_goals", "total_assists",
                 "total_minutes", "total_yellow_cards", "total_red_cards",
                 "goals_per_90", "assists_per_90"]
    for col in stat_cols:
        players[col] = players[col].fillna(0)

    # ------------------------------------------------------------------
    # 6. Encode categoricals
    # ------------------------------------------------------------------
    print("\n[5/5] Encoding categorical features ...")
    label_mappings: dict = {}

    CAT_COLS = ["position", "sub_position", "foot",
                "country_of_citizenship", "country_of_birth",
                "competition_type"]
    for col in CAT_COLS:
        if col in players.columns:
            players[col] = encode_categorical(players, col, label_mappings)

    # Drop remaining string columns that are not useful features
    # (free-text club names etc.)
    STRING_DROPS = ["name", "current_club_name", "current_club_id"]
    players.drop(columns=[c for c in STRING_DROPS if c in players.columns],
                 inplace=True)

    # ------------------------------------------------------------------
    # 7. Final cleanup
    # ------------------------------------------------------------------
    before = len(players)
    players.dropna(subset=["market_value_in_eur", "age", "position"],
                   inplace=True)
    print(f"\n  Rows before final dropna: {before:,}")
    print(f"  Rows after  final dropna: {len(players):,}")

    # Fill any remaining numeric NaNs with column medians
    num_cols = players.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if col != "market_value_in_eur":
            median_val = players[col].median()
            players[col] = players[col].fillna(median_val)

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    out_features = PROCESSED / "features.csv"
    out_encodings = PROCESSED / "label_encodings.json"

    players.to_csv(out_features, index=False)
    with open(out_encodings, "w", encoding="utf-8") as f:
        json.dump(label_mappings, f, indent=2, ensure_ascii=False)

    print()
    print(f"  ✓ Saved features:   {out_features}  ({len(players):,} rows, {len(players.columns)} cols)")
    print(f"  ✓ Saved encodings:  {out_encodings}")
    print()
    print("  Feature columns:", list(players.columns))
    print()
    print("  Next step: python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
