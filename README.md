# 📈 Pairs Trading Strategy — S&P 500

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Universe](https://img.shields.io/badge/Universe-S%26P%20500-orange)

A rigorous, end-to-end **statistical arbitrage pairs trading strategy** applied to the S&P 500 equity universe. Built in Python with a strict train/test split, Engle-Granger cointegration testing, OLS-derived hedge ratios, Ornstein-Uhlenbeck mean-reversion calibration, and a 225-combination parameter grid search — all net of transaction costs.
Live dashboard → [pairs-dashboard link here](https://megsaxena.github.io/pairs-trading-dashboard/)
Interactive simulator across all 21 cointegrated pairs and 4,725 parameter combinations.
---

## 🗂 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

Pairs trading exploits the long-run equilibrium between two co-integrated securities. When their price relationship temporarily diverges beyond a statistically meaningful threshold, the strategy goes long the underperformer and short the outperformer — then unwinds when the spread reverts.

This implementation improves on standard textbook treatments in several key ways:

| Standard Approach | This Implementation |
|---|---|
| Correlation only | Engle-Granger cointegration test |
| Price ratio as hedge | OLS log-price regression (proper β) |
| Fixed z-score window | OU half-life calibration (pair-specific) |
| No look-ahead control | Hard train/test split — no data leakage |
| Ignores costs | 10 bps/leg transaction costs applied |
| Single parameter set | 225-combination grid search |
| Sharpe not annualised | Annualised Sharpe (×√252) throughout |
| No multiple-testing correction | Benjamini-Hochberg FDR available |

---

## Results

Out-of-sample backtest: **March 2022 – March 2024** (503 trading days)

### Top Pairs

| Pair | Sector | Sharpe | Total Return | Max DD | Calmar | Half-life |
|---|---|---|---|---|---|---|
| **BAC / PNC** | Financials | **1.93** | +50.1% | -6.6% | 7.56 | 10.7 days |
| **DAL / NCLH** | Cross-sector | **1.80** | +58.1% | -6.7% | 8.68 | 11.8 days |
| **RJF / SCHW** | Financials | 1.53 | +58.3% | -13.3% | 4.38 | 13.6 days |
| **REG / FRT** | Real Estate | 0.62 | +10.1% | -10.1% | — | 5.6 days |

> All metrics are **out-of-sample** and **net of 40 bps round-trip transaction costs**. The test window spans the Fed's most aggressive hiking cycle in four decades — a deliberate stress test.

### BAC / PNC — Best Configuration

```
Entry: 1.75σ  |  Exit: 0.75σ  |  Stop: 4.0σ  |  Window: 30 days
Sharpe: 1.93  |  Return: 50.1%  |  Max DD: -6.6%  |  Win Rate: 25.5%
```

![BAC/PNC Chart](pair_BAC_PNC.png)

### DAL / NCLH — Cross-Sector Highlight

```
Entry: 2.50σ  |  Exit: 0.25σ  |  Stop: 4.0σ  |  Window: 90 days
Sharpe: 1.80  |  Return: 58.1%  |  Max DD: -6.7%  |  Calmar: 8.68
```

> Delta Air Lines and Norwegian Cruise Line are in different GICS sectors but share a deep macro dependency: both are high-leverage leisure-travel businesses exposed to the same consumer cycle and fuel costs. This pair would be missed entirely by a same-sector-only screen.

![DAL/NCLH Chart](pair_DAL_NCLH.png)

---

## Methodology

The pipeline runs in seven sequential stages. All parameters are estimated **in-sample only** and applied forward.

```
S&P 500 Universe (488 tickers)
        │
        ▼
[1] Correlation Screen
    Pearson > 0.70 on in-sample log-returns
    Top 300 pairs selected
        │
        ▼
[2] Cointegration Test (Engle-Granger)
    ADF on OLS residuals · p < 0.05
    Top 100 by ascending p-value
        │
        ▼
[3] Hedge Ratio Estimation
    OLS: log(S₁) ~ β·log(S₂)
    Spread = log(S₁) − β·log(S₂)
        │
        ▼
[4] OU Half-Life Calibration
    Δsₜ = θ·sₜ₋₁ + εₜ
    τ½ = −ln(2)/θ
    Valid range: 5–120 days
        │
        ▼
[5] Z-Score Signal Generation
    Window = 2 × half-life (pair-specific)
    Enter: |z| > entry threshold
    Exit:  |z| < exit threshold
    Stop:  |z| > stop threshold
        │
        ▼
[6] Grid Search (225 combinations)
    Entry: {1.50, 1.75, 2.00, 2.25, 2.50}σ
    Exit:  {0.25, 0.50, 0.75}σ
    Stop:  {3.0, 3.5, 4.0}σ
    Window:{20, 30, 40, 60, 90} days
        │
        ▼
[7] Performance Evaluation
    Net of 10 bps/leg transaction costs
    Metrics: Sharpe · MaxDD · Calmar · Win Rate · Profit Factor
```

### Train / Test Split

| Period | Dates | Purpose |
|---|---|---|
| **In-sample** | Feb 2020 – Mar 2022 | Pair selection, β estimation, half-life calibration |
| **Out-of-sample** | Mar 2022 – Mar 2024 | Signal generation, P&L, all reported metrics |

---

## Project Structure

```
pairs-trading/
│
├── pairs_trading.py          # Main strategy pipeline
├── pairs_grid_results.xlsx   # Grid search results (4,725 combinations)
├── dashboard.html            # Interactive web dashboard
├── README.md
│
├── charts/
│   ├── pair_BAC_PNC.png
│   ├── pair_DAL_NCLH.png
│   ├── pair_REG_FRT.png
│   └── portfolio_pnl.png
│
└── paper/
    └── pairs_trading_paper.docx   # Full research paper
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/yourusername/pairs-trading.git
cd pairs-trading

# Install dependencies
pip install yfinance statsmodels pandas numpy matplotlib openpyxl
```

---

## Usage

Run the full pipeline end-to-end:

```bash
python pairs_trading.py
```

This will:
1. Download S&P 500 price data from Yahoo Finance
2. Screen for correlated pairs (top 300 by Pearson correlation)
3. Test cointegration (Engle-Granger, top 100 by p-value)
4. Estimate OLS hedge ratios and OU half-lives
5. Run the 225-combination parameter grid search
6. Save results to `W:\path\to\your\output\folder` (configurable in `CFG`)
7. Generate per-pair charts and a portfolio equity curve

**Outputs:**
- `pairs_grid_results.xlsx` — full grid search results, formatted with colour-coded performance table
- `pair_XX_YY.png` — chart per top pair (spread, z-score, P&L)
- `portfolio_pnl.png` — equal-weight portfolio equity curve

---

## Configuration

All key parameters are set in the `CFG` and `GRID` dictionaries at the top of `pairs_trading.py`:

```python
CFG = dict(
    start_date          = "2020-02-11",
    train_end_date      = "2022-03-14",   # end of in-sample period
    test_end_date       = "2024-03-14",   # end of out-of-sample period
    corr_threshold      = 0.70,           # minimum return correlation
    coint_pval          = 0.05,           # Engle-Granger significance level
    fdr_correction      = False,          # Benjamini-Hochberg correction
    same_sector_only    = False,          # False = full cross-sector universe
    max_corr_pairs      = 300,            # pairs to screen for cointegration
    top_coint_pairs     = 100,            # pairs to pass to calibration
    transaction_cost    = 0.0010,         # per-leg cost (10 bps)
    min_halflife_days   = 5,              # minimum OU half-life
    max_halflife_days   = 120,            # maximum OU half-life
    n_workers           = 4,              # parallel workers for coint tests
    top_n_plot          = 5,             # pairs to chart
)

GRID = dict(
    entry_thresholds = [1.5, 1.75, 2.0, 2.25, 2.5],
    exit_thresholds  = [0.25, 0.5, 0.75],
    z_windows        = [20, 30, 40, 60, 90],
    stop_thresholds  = [3.0, 3.5, 4.0],
)
```

---

## Dashboard

An interactive dashboard is included (`dashboard.html`) — no server required.

**Features:**
- Select from all 21 cointegrated pairs
- Adjust entry/exit/stop/window parameters with sliders
- Live metric cards: Sharpe, return, max drawdown, Calmar, win rate
- Simulated P&L chart and Sharpe vs drawdown scatter
- All-pairs leaderboard ranked by best Sharpe

**To use locally:** open `dashboard.html` in any browser.

---

## Limitations

- **Survivorship bias** — uses current S&P 500 membership; historically removed stocks are excluded
- **Regime breaks** — REG/FRT demonstrates spread drift during the 2022 rate shock; a production system needs rolling cointegration monitoring
- **Static hedge ratio** — β is fixed at the in-sample estimate; a Kalman filter would allow real-time updating
- **Capacity constraints** — market impact not modelled; alpha erodes with position size
- **No portfolio-level risk** — pairs are treated independently; factor exposure aggregation required for live trading

---

## References

- Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative value arbitrage rule. *Review of Financial Studies*, 19(3), 797–827.
- Avellaneda, M., & Lee, J. H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*, 10(7), 761–782.
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance.
- Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the theory of Brownian motion. *Physical Review*, 36(5), 823–841.

---

## Author

**Meghna Saxena** — Junior Portfolio Manager, QC Partners GmbH

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/meghnasaxena-/)

---

*Built as a quantitative research portfolio project. All backtests are out-of-sample and net of transaction costs. This is not financial advice.*
