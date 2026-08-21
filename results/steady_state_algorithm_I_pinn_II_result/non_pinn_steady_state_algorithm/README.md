# Non-PINN IAQ Parameter Estimation

This module implements an algorithmic, non-PINN approach for estimating piecewise-constant ventilation rate \(Q\) and indoor source rate \(S\) from CO₂ time-series data.

The method is based on the single-zone mass-balance model:

\[
V\frac{dC}{dt}=Q(C_{out}-C)+S
\]

with analytical solution

\[
C(t)=C_{ss}+(C_0-C_{ss})e^{-t/\tau},
\]

where

\[
C_{ss}=C_{out}+\frac{S}{Q},
\qquad
\tau=\frac{V}{Q}.
\]

## Pipeline

The pipeline consists of two main stages followed by evaluation:

```text
Stage 1: Segmentation
    ├── smooth derivative
    ├── steady/transient labeling
    ├── initial segment cutting
    └── changepoint refinement

Stage 2: Parameter estimation
    ├── estimate Css
    ├── fit tau
    ├── calculate Q
    └── calculate S

Evaluation
    ├── compare with truth
    └── diagnostic plots