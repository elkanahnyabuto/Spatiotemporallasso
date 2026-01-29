# Estimation of Spatiotemporal Autoregressive Effects using Lasso

## Overview

This repository implements a Spatiotemporal Autoregressive model with joint estimation of regression coefficients, temporal dependence, and a spatial weights matrix using penalised maximum likelihood.

## Model Formulation

### Spatiotemporal Autoregressive Model

Assume that the spatiotemporal panel is drawn from a univariate random process $\{Y_t(\boldsymbol{s}): t = 1, \ldots, T; \boldsymbol{s} \in D_s\}$ with a finite set $D_s$ of $n$ locations, $\boldsymbol{s_1}, \ldots, \boldsymbol{s_n} $. Moreover, let $\mathbf{Y_t}$ be the stacked vector of all random variables across space at time point $t$, i.e.,  $\boldsymbol{Y_t} = (Y_t(\boldsymbol{s_1}), \ldots, Y_t(\boldsymbol{s_n}))'$. We consider data that follows a spatiotemporal dynamic panel data model 

$$\boldsymbol{Y_t} = \boldsymbol{X_t}\boldsymbol{\beta} + \sum_{p = 1}^{P} \mathbf{\Phi}_p \boldsymbol{Y_{t-p}} + \mathbf{W} \boldsymbol{Y_t} + \boldsymbol{\varepsilon_t} $$

which further simplifies to

$$ \boldsymbol{Y_t} = (\mathbf{I} - \mathbf{W})^{-1}\left(\mathbf{X}_t \boldsymbol{\beta} + \mathbf{\Phi}_p \boldsymbol{Y_{t-1}}+\varepsilon_t\right),%\qquad\varepsilon_t \sim \mathcal{N}(0, \sigma^2 I_n).$$

where
- $\mathbf{Y}_t \in \mathbb{R}^n$ is the response vector at time $t$
- $X_t \in \mathbb{R}^{n \times k}$ is the matrix of covariates
- $\beta \in \mathbb{R}^k$ is the vector of regression coefficients
- $\Phi_p = \mathrm{diag}(\phi_p(s_1), \dots, \phi_p(s_n))$ are temporal model coefficients for the previous  $p$ realisations of $\boldsymbol{Y_t}$ at time $t$
- $\mathbf{W} \in \mathbb{R}^{n \times n}$ is an unknown spatial weights matrix
- $\sigma^2$ is the error term


### Penalised Log-Likelihood

Parameters are estimated by maximising the penalised log-likelihood

$$\ell(\theta)= T \ln | I - W |-\frac{nT}{2} \ln (2\pi\sigma^2)-\frac{1}{2\sigma^2}\sum_{t=2}^{T}\left\(\mathbf{Y}_t-W \mathbf{Y}_t-X_t \beta-\Phi \mathbf{Y}_{t-1}\right\)^T \left\(\mathbf{Y}_t-W \mathbf{Y}_t-X_t \beta-\Phi \mathbf{Y}_{t-1}\right\)-\left(\lambda_1 \sum_{i,j = 1}^{n} |w_{ij}| + \lambda_2\sum_{i = 1}^{n} |\phi_1(s_i)| + \lambda_3\sum_{k = 1}^{k} |\beta_l| \right),$$

where $\lambda_1,\lambda_2 \quad \text{and}\quad \lambda_3)$ are the $l_1$ penalties associated with the weights,temporal coefficients and regressors respectively.  Each set of parameters is subjected to an $l_1$ penalty due to its capacity to induce sparsity in parameter estimates.

The spatial weights matrix $\mathbf{W}$ is estimated subject to the constraints ensure model stability and interpretability of spatial dependence.

$$ \left|(\mathbf{I} - \mathbf{W})^{-1}\mathbf{\Phi}_1\right|_\infty < 1 \quad \quad \text{if} \; \; 0\leq \phi_1(s_i) \leq 1\; \, \; \forall i \quad \text{and} \quad \sum_{j=1}^{n} w_{ij} \leq 1,\quad w_{ii} = 0,\quad \forall i = 1, \dots, n.$$


