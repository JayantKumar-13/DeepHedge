# DeepHedge: Dynamic Delta Hedging for NIFTY Options using Deep Learning

A quantitative finance and deep learning project that implements **dynamic option hedging** for NIFTY options using **Geometric Brownian Motion (GBM)** simulations, **Black-Scholes Greeks**, **Implied Volatility**, and a **Deep Hedging neural network** optimized using **Conditional Value at Risk (CVaR)**.

The project demonstrates how deep learning can learn an optimal hedging strategy that minimizes portfolio tail risk under simulated market conditions.

---

## Project Overview

This project builds an end-to-end deep hedging pipeline for NIFTY options using minute-level market data.

The workflow includes:

- Loading real NIFTY option chain data for **expiry** and **non-expiry** trading days.
- Calculating **Implied Volatility** from observed option prices.
- Computing **Black-Scholes Greeks** (Delta, Gamma, Theta).
- Simulating multiple future asset price paths using **Geometric Brownian Motion (GBM)**.
- Training a **Deep Hedging Model** using TensorFlow.
- Optimizing the hedge using **CVaR Loss** to reduce downside risk.
- Comparing hedge positions and portfolio performance through visualizations.

---

## Repository Structure

```text
deep-hedging-nifty-options/
│── README.md
│── requirements.txt
│
├── data/
│   ├── 20260204_option_minute_prices_non_expiry.csv
│   └── 20260205_option_minute_prices_expiry.csv
│
├── notebooks/
│   └── options_trading_price_prediction.ipynb
│
├── images/
│   ├── Simulated_GBM_Price_Paths.png
│   ├── Delta_Hedging_Results.png
│   ├── Delta_vs_Strike_Price.png
│   ├── Gamma_vs_Strike_Price.png
│   ├── Theta_vs_Strike_Price.png
│   ├── Implied_Volatility_vs_Strike_Price.png
│   └── P&L_Distribution.png
```

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **SciPy**
- **Matplotlib**
- **Black-Scholes Option Pricing Model**

---

## Dataset

The project uses minute-level **NIFTY option chain data** for two trading sessions.

| Dataset | Description |
|---------|-------------|
| `20260205_option_minute_prices_expiry.csv` | NIFTY options data collected on expiry day. |
| `20260204_option_minute_prices_non_expiry.csv` | NIFTY options data collected on a non-expiry trading day. |

The datasets contain:

- Date
- Minute timestamp
- Option/Futures symbol
- Last traded price
- Strike price (parsed from symbol)
- Option type (Call / Put)

---

## Workflow

1. Load minute-level option chain data.
2. Filter option contracts for a specific observation time.
3. Extract underlying futures price.
4. Compute **Implied Volatility** using Black-Scholes inversion.
5. Calculate **Delta, Gamma, and Theta** for each option.
6. Generate multiple GBM price paths.
7. Train a Deep Hedging neural network using simulated price paths.
8. Optimize hedging strategy using **CVaR Loss**.
9. Evaluate hedge performance using portfolio P&L distribution.

---

## Model

### Inputs

- Underlying asset price
- Black-Scholes Delta
- Previous hedge position

### Output

- Predicted hedge ratio (Delta)

### Loss Function

The model minimizes **Conditional Value at Risk (CVaR)** of the hedged portfolio P&L instead of traditional mean squared error.

---

## Results & Visualizations

### 1. Simulated GBM Price Paths

The figure below shows **10 simulated Geometric Brownian Motion (GBM)** price paths generated using the underlying NIFTY futures price. These simulated trajectories are used as training paths for the deep hedging model.

![Simulated GBM Price Paths](images/Simulated_GBM_Price_Paths.png)

---

### 2. Deep Hedging Results

This visualization summarizes the learned hedging strategy throughout the trading session.

It includes:
- **Delta hedge position** over time.
- **Implied volatility** evolution.
- **Underlying NIFTY futures price** movement.

![Deep Hedging Results](images/Delta_Hedging_Results.png)

---

### 3. Delta vs Strike Price

Black-Scholes Delta for Call and Put options across different strike prices at **11:00 AM**.

![Delta vs Strike Price](images/Delta_vs_Strike_Price.png)

---

### 4. Gamma vs Strike Price

Gamma variation across strike prices for Call and Put options.

![Gamma vs Strike Price](images/Gamma_vs_Strike_Price.png)

---

### 5. Theta vs Strike Price

Theta decay across strike prices for Call and Put options.

![Theta vs Strike Price](images/Theta_vs_Strike_Price.png)

---

### 6. Implied Volatility vs Strike Price

Implied volatility smile observed across NIFTY option strike prices at **11:00 AM**.

![Implied Volatility vs Strike Price](images/Implied_Volatility_vs_Strike_Price.png)

---

### 7. Portfolio P&L Distribution

Distribution of hedged portfolio profit and loss after applying the Deep Hedging strategy and comparison with Black-Scholes Delta Hedging.

![Portfolio P&L Distribution](images/P&L_Distribution_for_Deep_Hedging_vs_Black_Scholes_Delta_Hedging.png)

---

### Portfolio P&L Distribution

![P&L Distribution](images/P&L_Distribution.png)

---

## Key Concepts Implemented

- Black-Scholes Option Pricing
- Implied Volatility Estimation
- Option Greeks (Delta, Gamma, Theta)
- Geometric Brownian Motion (GBM)
- Dynamic Delta Hedging
- Deep Hedging using Neural Networks
- CVaR (Conditional Value at Risk) Optimization

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deep-hedging-nifty-options.git
cd deep-hedging-nifty-options
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Run the notebook in **Google Colab** or **Jupyter Notebook**.

```bash
jupyter notebook notebooks/options_trading_price_prediction.ipynb
```

---

## Future Improvements

- Support multiple option expiries.
- Train on historical multi-day option chain data.
- Compare Deep Hedging with classical Delta Hedging.
- Extend the framework to include transaction costs and volatility smile dynamics.