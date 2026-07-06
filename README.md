# Decision-Centric Inventory Allocation Platform

> *Given a forecast, a budget, and current inventory — where should the next
> shipment go, and how confident should we be?*

This is a decision-support tool, not a prediction pipeline. Every module
exists to feed, explain, or act on one final output: a shipment allocation
a manager could actually approve.

---

## 1. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the demo dataset (synthetic, shaped like the Walmart
#    Recruiting "Store Sales Forecasting" dataset — swap in the real CSVs
#    under data/raw/ and nothing downstream changes)
python data/generate_synthetic_data.py

# 3. Run the full analytical pipeline end-to-end, module by module
python run_pipeline.py

# 4. Launch the dashboard
streamlit run app/dashboard.py

# 5. Or launch the API
uvicorn app.api:app --reload --port 8000
```

Every module also runs standalone for debugging:
`python src/module3_forecasting.py`, etc.

---

## 2. Architecture

```
Raw Retail Data (Sales, Inventory, Stores, Calendar)
                    │
        Module 1 — Validate + Clean + Merge + Feature Engineer
                    │
       ┌────────────┴────────────┐
       │                         │
Module 3 — Demand Forecast   Module 4 — Store Segmentation
(+ prediction interval)       (behavior clusters)
       │                         │
       └────────────┬────────────┘
                    │
     Module 5 — Confidence-Weighted Inventory Optimization  ◄── the centerpiece
                    │
     Module 6 — Explainability (SHAP / permutation fallback)
                    │
     Module 7 — LLM Business Consultant (templated, grounded queries)
                    │
        Module 9 — Streamlit Dashboard  +  FastAPI + Docker
                    │
        Module 8 — MLflow tracking + rolling-MAE monitoring (runs alongside)
```

Module 2 (EDA + KPIs) sits directly on top of Module 1's output and feeds
the dashboard's Overview page and Module 7's executive-summary workflow —
it's a set of read-only analyses, not a distinct pipeline stage, which is
why it isn't drawn as its own box above.

---

## 3. Project layout

```
inventory-platform/
├── data/
│   ├── generate_synthetic_data.py   # stand-in for the real dataset
│   ├── raw/                          # sales.csv, stores.csv, calendar.csv
│   └── processed/                    # feature_table.parquet (Module 1 output)
├── src/
│   ├── module1_data_foundation.py
│   ├── module2_eda_kpis.py
│   ├── module3_forecasting.py
│   ├── module4_segmentation.py
│   ├── module5_optimization.py       # the centerpiece
│   ├── module6_explainability.py
│   ├── module7_llm_layer.py
│   └── module8_monitoring.py
├── app/
│   ├── dashboard.py                  # Streamlit: 5 pages
│   └── api.py                        # FastAPI: /forecast /optimize /explain /ask
├── run_pipeline.py                   # runs modules 1-8 end-to-end with printed trace
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 4. The centerpiece: confidence-weighted allocation (Module 5)

The optimizer does not treat every forecast equally. A wide prediction
interval (low confidence) pulls the demand estimate fed into the LP down
toward the interval's lower bound; a tight interval (high confidence)
allocates near the full point forecast.

**The exact formula**:

```
effective_demand = lower_bound + confidence_score × (forecast − lower_bound)
```

- `confidence_score = 1` → `effective_demand = forecast` (tight interval, allocate to the point estimate)
- `confidence_score = 0` → `effective_demand = lower_bound` (wide interval, allocate conservatively)
- Anything in between is linear interpolation — no arbitrary safety-factor constant.

```
Store A → Forecast: 1,000 | Confidence: 55% (wide interval)
         → effective_demand shrinks toward the lower bound (conservative)

Store B → Forecast: 950   | Confidence: 95% (tight interval)
         → effective_demand stays close to 950 (aggressive)
```

`confidence_score` itself comes from Module 3: it's a normalized inverse
of the prediction interval's width relative to the point forecast — a
wider interval (as a share of the forecast) produces a lower score.

**The LP**, built as a single parameterized function
(`optimize_allocation` in `module5_optimization.py`) so scenario analysis
is just "call it again with different arguments":

- **Decision variable:** units allocated to each store, `x_i`
- **Minimize:** holding cost + transport cost (per-km × distance) + a
  linearized lost-sale penalty on any unmet effective demand
- **Subject to:** per-store capacity, total warehouse capacity, and an
  optional budget cap
- **Solver:** Google OR-Tools (GLOP)

Scenario analysis (`run_scenario` in the same file) re-runs this function
with a changed `CostParams` — a 10% budget cut, more warehouse capacity, a
demand-spike assumption — and is the highest-ROI feature in the whole
project because it turns "here's a forecast" into "here's a tool you can
actually negotiate with."

---

## 5. Module-by-module notes

| Module | Key design decision | Why |
|---|---|---|
| 1. Data Foundation | Lag/rolling features use `shift(1)` before rolling | Prevents the model from peeking at its own target — the most common and easiest-to-miss leak in time-series projects |
| 2. EDA + KPIs | Welch's t-test (not Student's) for the promo effect | Doesn't assume equal variance between promo/non-promo weeks |
| 3. Forecasting | Walk-forward validation, not random split | Random splits let the model train on future weeks relative to its test week — inflates reported accuracy |
| 3. Forecasting | Prediction intervals via residual bootstrap | Distribution-free; doesn't assume Gaussian residuals; easy to explain without hand-waving |
| 4. Segmentation | KMeans on behavioral features, not just store_type | "High-volume/low-volatility" is a much more actionable label than "Type A store" |
| 5. Optimization | Single parameterized function | Everything else (scenario analysis) is free once this is true |
| 6. Explainability | Top 3-4 drivers only, not a full SHAP dump | A manager needs the reason, not a feature-importance research paper |
| 7. LLM layer | Fixed templates over structured JSON, not open chat | The LLM explains numbers it was handed — it never invents or looks anything up |
| 8. Monitoring | Rolling MAE vs. one threshold — nothing fancier | Timeboxed on purpose; a drift-detection framework wasn't the point of this project |

---

## 6. Tech stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels/SciPy |
| Explainability | SHAP |
| Optimization | Google OR-Tools |
| Experiment tracking | MLflow |
| Generative AI | Anthropic API, structured/templated prompting |
| Backend | FastAPI |
| Frontend | Streamlit + Plotly |
| Deployment | Docker, Render/Railway |

---
