# ⚽ Football Player Transfer Value Predictor

Predict a football player's estimated Transfermarkt market value using a
**Random Forest regression model** trained on 50,000+ real player profiles.

> **Important:** This tool predicts *estimated market value* — Transfermarkt's
> editorial valuation.  This is **not** an actual transfer fee prediction.

---

## 📂 Project Structure

```
football_transfer_predictor/
├── app.py                    # Streamlit dashboard (main entry point)
├── requirements.txt          # Python dependencies
├── README.md
│
├── src/
│   ├── download_data.py      # Downloads raw CSVs from dcaribou/transfermarkt-datasets
│   ├── preprocess.py         # Feature engineering and cleaning
│   ├── train.py              # Model training and evaluation
│   └── predictor.py          # Inference wrapper used by the dashboard
│
├── data/
│   ├── raw/                  # Raw CSVs (created by download_data.py)
│   └── processed/            # features.csv, label_encodings.json
│
├── models/
│   ├── rf_pipeline.joblib    # Trained Random Forest pipeline
│   ├── feature_meta.json     # Feature column order
│   ├── metrics.json          # Test-set evaluation results
│   └── feature_importance.csv
│
└── assets/                   # Static assets (optional)
```

---

## 🚀 Setup and Usage

### 1. Prerequisites

- **Python 3.10, 3.11, or 3.12** — download from https://www.python.org/downloads/
  - On Windows: tick **"Add Python to PATH"** during installation.
- **Internet connection** — required for the first data download (~150 MB).

### 2. Install dependencies

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python src/download_data.py
```

This downloads four CSV files from the publicly hosted
[dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
R2 bucket and places them in `data/raw/`.

**If automatic download fails**, download manually:

| File | URL |
|------|-----|
| players.csv.gz | https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/players.csv.gz |
| appearances.csv.gz | https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/appearances.csv.gz |
| clubs.csv.gz | https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/clubs.csv.gz |
| competitions.csv.gz | https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/competitions.csv.gz |

Decompress each `.gz` file and place the `.csv` files in `data/raw/`.

### 4. Preprocess the data

```bash
python src/preprocess.py
```

Outputs `data/processed/features.csv` and `data/processed/label_encodings.json`.
Expected runtime: ~1–3 minutes.

### 5. Train the model

```bash
python src/train.py
```

Trains a **median baseline** and a **Random Forest** model, prints test-set
metrics (MAE, RMSE, R²), and saves the pipeline to `models/`.
Expected runtime: ~3–8 minutes depending on CPU.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Model Details

| | Baseline (Median) | Random Forest |
|---|---|---|
| Algorithm | DummyRegressor | RandomForestRegressor (300 trees) |
| Target | log₁(market_value_in_eur) | same |
| Train/Test split | 80 / 20 | 80 / 20 |

**Features used (after preprocessing):**

| Feature | Type | Notes |
|---------|------|-------|
| age | numeric | derived from date_of_birth |
| position | encoded int | Attack / Midfield / Defender / Goalkeeper |
| sub_position | encoded int | granular position |
| foot | encoded int | right / left / both |
| height_in_cm | numeric | |
| country_of_citizenship | encoded int | |
| country_of_birth | encoded int | |
| competition_type | encoded int | domestic_league / cup / international |
| total_appearances | numeric | last 2 seasons |
| total_goals | numeric | last 2 seasons |
| total_assists | numeric | last 2 seasons |
| total_minutes | numeric | last 2 seasons |
| total_yellow_cards | numeric | last 2 seasons |
| total_red_cards | numeric | last 2 seasons |
| goals_per_90 | numeric | derived |
| assists_per_90 | numeric | derived |
| squad_size | numeric | from clubs.csv |
| national_team_players | numeric | from clubs.csv |

**Leakage prevention:** `highest_market_value_in_eur` is explicitly excluded —
it would encode the target indirectly and inflate accuracy artificially.

---

## ⚠️ Known Limitations

1. **Market value ≠ transfer fee.** Transfermarkt values are editorial estimates.
2. **Recency lag.** Data is current to July 2026; very recent signings and injuries are not captured.
3. **Regression to the mean.** Random Forests under-predict elite player values (€100M+) and may over-predict fringe players.
4. **No injury or contract history features.** Contract length, injury frequency, and release clauses are not modelled.
5. **Club prestige approximation.** Only squad size and international player count are used as club-quality proxies.

---

## 📄 Data Attribution

Data sourced from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets),
published under the community spirit of open football data.
Original data © [Transfermarkt GmbH & Co. KG](https://www.transfermarkt.com/).
**Non-commercial use only.**

---

## 🛠 Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not found | Install Python from python.org; tick "Add to PATH" |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| `features.csv not found` | Run `python src/preprocess.py` |
| `rf_pipeline.joblib not found` | Run `python src/train.py` |
| Download fails (firewall) | Download CSVs manually (see step 3 above) |
| Streamlit port busy | `streamlit run app.py --server.port 8502` |
