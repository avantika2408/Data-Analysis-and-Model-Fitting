# Bayesian Signal Extraction Using MCMC

This project demonstrates Bayesian inference techniques for signal extraction in high-energy physics data using Markov Chain Monte Carlo (MCMC) sampling.

## 📌 Objective
To perform parameter estimation of a double Gaussian peak over a linear Chebyshev background using MCMC and visualize the posterior distributions.

---

## 🧪 Problem Description

In many particle physics experiments, signal peaks are superimposed on complex backgrounds. This project models a dataset composed of:

- Two **Gaussian peaks** representing signal processes
- A **Chebyshev polynomial** representing the background

The goal is to accurately estimate the parameters (means, widths, amplitudes, background coefficients) using **Bayesian inference**.

---

## 🛠 Tools and Libraries

- **Cobaya** – Bayesian sampling using MCMC
- **GetDist** – Visualization of marginal posteriors and correlations
- **NumPy**, **Matplotlib** – Data handling and plotting
- **SciPy** – Chebyshev background modeling
- **Python** – Core language

---


