# Post-Mortem: Why Noisy Sphere Data Broke the IFT Solver

*A mathematical analysis of regularisation washout, weight explosion, and why the Debye form factor is a kinder test fixture.*

---

## 1. The SAXS Indirect Fourier Transform Problem

The goal is to recover the pair distance distribution $P(r)$ from measured scattering intensity $I(q)$. The exact relationship is

$$
I(q) = 4\pi \int_0^{D_{\max}} P(r)\, \frac{\sin(qr)}{qr}\, dr
$$

Discretising over $N_q$ measured points $\{q_i\}$ and $N_r$ reconstruction nodes $\{r_j\}$ gives the linear system

$$
\mathbf{I} = \mathbf{K}\,\mathbf{p}, \qquad K_{ij} = 4\pi\,\frac{\sin(q_i r_j)}{q_i r_j}\,\Delta r_j
$$

This is a **Fredholm integral equation of the first kind** — it is ill-posed: small perturbations in $\mathbf{I}$ can produce arbitrarily large oscillations in $\mathbf{p}$. Regularisation is essential.

---

## 2. The Regularised Objective

We minimise the combined objective

$$
\mathcal{L}(\mathbf{p}) = \chi^2(\mathbf{p}) + \lambda\,\mathcal{R}(\mathbf{p})
$$

### 2.1 The Data Term

$$
\chi^2 = \sum_{i=1}^{N_q} \frac{\bigl(I_{\mathrm{obs}}(q_i) - [\mathbf{K}\mathbf{p}]_i\bigr)^2}{\sigma_i^2}
= \bigl(\mathbf{W}^{1/2}(\mathbf{I}_{\mathrm{obs}} - \mathbf{K}\mathbf{p})\bigr)^\top \bigl(\mathbf{W}^{1/2}(\mathbf{I}_{\mathrm{obs}} - \mathbf{K}\mathbf{p})\bigr)
$$

where $\mathbf{W} = \mathrm{diag}(1/\sigma_i^2)$ is the **weight matrix**.

### 2.2 The Regularisation Term

Using a second-derivative (smoothness) penalty with matrix $\mathbf{L}$:

$$
\mathcal{R}(\mathbf{p}) = \|\mathbf{L}\mathbf{p}\|^2 = \mathbf{p}^\top \mathbf{H}\,\mathbf{p}, \qquad \mathbf{H} = \mathbf{L}^\top\mathbf{L}
$$

### 2.3 The Normal Equations

Setting $\nabla_{\mathbf{p}}\,\mathcal{L} = 0$:

$$
\boxed{\bigl(\mathbf{K}^\top \mathbf{W} \mathbf{K} + \lambda\,\mathbf{H}\bigr)\,\mathbf{p} = \mathbf{K}^\top \mathbf{W}\,\mathbf{I}_{\mathrm{obs}}}
$$

Let $\mathbf{A} = \mathbf{K}^\top \mathbf{W} \mathbf{K}$ and $\mathbf{b} = \mathbf{K}^\top \mathbf{W}\,\mathbf{I}_{\mathrm{obs}}$.
The solution is $\mathbf{p} = (\mathbf{A} + \lambda\mathbf{H})^{-1}\mathbf{b}$.

The scalar $\lambda$ controls the trade-off: large $\lambda$ → smooth $P(r)$; small $\lambda$ → fits noise.

---

## 3. The Noise Model

For our synthetic data we used a **proportional noise model**:

$$
\sigma(q) = \frac{I(q)}{k}, \qquad k > 0
$$

So $k = 5$ means roughly 20% relative uncertainty at every point. The weights become

$$
W_{ii} = \frac{1}{\sigma_i^2} = \frac{k^2}{I(q_i)^2}
$$

This is physically motivated: real photon-counting detectors have Poisson statistics, so $\sigma \propto \sqrt{I}$ (not $I$). A purely proportional model is actually *worse* than Poisson at low $I$ — the weight blows up even faster. But it is a common simplification and perfectly fine when $I(q)$ has no zeros. That last clause is the key.

---

## 4. The Sphere Form Factor and Its Fatal Zeros

The sphere form factor is

$$
I_{\text{sphere}}(q) = \left[3\,\frac{\sin(qR) - qR\cos(qR)}{(qR)^3}\right]^2
$$

This has **true zeros** wherever $\tan(qR) = qR$, i.e. at

$$
qR \approx 4.493,\; 7.725,\; 10.904,\;\ldots
$$

At those $q$ values, $I(q) = 0$ exactly, so

$$
\sigma(q) = \frac{I(q)}{k} = 0 \implies W_{ii} = \frac{1}{\sigma_i^2} \to \infty
$$

---

## 5. The Washout: Why $\lambda$ Becomes Irrelevant

### 5.1 Scale separation at the zeros

Near a minimum $q^*$, Taylor-expand:

$$
I(q^* + \epsilon) \approx c\,\epsilon^2 + O(\epsilon^3) \implies \sigma \approx \frac{c}{k}\epsilon^2 \implies W \approx \frac{k^2}{c^2\,\epsilon^4}
$$

The weight **diverges quartically** as $\epsilon \to 0$.

### 5.2 Effect on the data matrix $\mathbf{A}$

The matrix $\mathbf{A} = \mathbf{K}^\top \mathbf{W} \mathbf{K}$ has entries

$$
A_{jl} = \sum_i K_{ij}\,W_{ii}\,K_{il} = \sum_i \frac{K_{ij}\,K_{il}}{\sigma_i^2}
$$

A single point near the zero with $W_{ii} \gg 1$ dominates this sum. The **largest eigenvalue** of $\mathbf{A}$ is approximately

$$
\mu_{\max}(\mathbf{A}) \;\sim\; W_{\max} \cdot \|{\mathbf{k}}_{\max}\|^2
$$

where $\mathbf{k}_{\max}$ is the row of $\mathbf{K}$ at the near-zero point. As $\sigma \to 0$ this eigenvalue diverges.

### 5.3 The effective regularisation parameter

Think about the solution in the eigenbasis of $\mathbf{A}$. In each eigendirection $v$ with eigenvalue $\mu$, the regularisation term $\lambda\,\mathbf{H}$ damps that direction only when $\lambda\,h \gtrsim \mu$ (where $h$ is the corresponding eigenvalue of $\mathbf{H}$). The **effective regularisation** in direction $v$ is

$$
\lambda_{\text{eff}} = \frac{\lambda\,h}{\mu + \lambda\,h} \approx \frac{\lambda\,h}{\mu} \quad \text{when } \mu \gg \lambda\,h
$$

When $\mu \to \infty$ (from the exploding weight), $\lambda_{\text{eff}} \to 0$. **No finite $\lambda$ can compensate.** Increasing $\lambda$ merely has to compete with an ever-larger $\mu$; the ratio stays vanishingly small.

In practice, to actually regularise the solution, one would need

$$
\lambda \;\gtrsim\; \frac{\mu_{\max}}{h_{\max}} \;\sim\; \frac{k^2}{c^2\,\epsilon^4\,\Delta r^2}
$$

which, near a true zero, requires $\lambda \to \infty$ — completely suppressing the data fit everywhere else.

### 5.4 A geometric view

The solution $\mathbf{p}^* = (\mathbf{A} + \lambda\mathbf{H})^{-1}\mathbf{b}$ lives in a high-dimensional space. The "data directions" (large eigenvalues of $\mathbf{A}$) are determined almost entirely by the two or three near-zero points. Those points demand $[\mathbf{K}\mathbf{p}]_{i^*} \approx 0$ with essentially infinite precision. Because the kernel rows $\mathbf{k}_{i^*}$ at those $q$ values are not zero, this imposes a hard constraint on $\mathbf{p}$ — one that has nothing to do with the true $P(r)$ and that regularisation cannot undo.

The solver is simultaneously trying to:
- (a) Force $\mathbf{K}\mathbf{p}$ to be near zero at the minima (infinite weight).
- (b) Fit $\mathbf{K}\mathbf{p}$ to the noisy data everywhere else.
- (c) Keep $\mathbf{p}$ smooth via $\lambda\mathbf{H}$.

Constraint (a) wins, distorting $P(r)$ far from the truth.

### 5.5 The right-hand side is also corrupted

$\mathbf{b} = \mathbf{K}^\top\mathbf{W}\mathbf{I}_{\mathrm{obs}}$. Near the zeros, $I_{\mathrm{obs}}(q_i^*) = I(q_i^*) + \eta_i$ where $\eta_i \sim \mathcal{N}(0, \sigma_i^2)$. Even a tiny residual noise $\eta_i$ at zero gets multiplied by $W_{ii} = 1/\sigma_i^2$, giving a contribution to $\mathbf{b}$ of order $\eta_i/\sigma_i^2 \sim 1/\sigma_i$, which diverges. The system is being driven toward the *noise*, not the *signal*.

---

## 6. Why the Reduced $\chi^2$ Doesn't Reveal This

After solving, we report

$$
\chi^2_{\mathrm{red}} = \frac{1}{N_q - N_r}\sum_i \frac{(I_{\mathrm{obs},i} - [\mathbf{K}\hat{\mathbf{p}}]_i)^2}{\sigma_i^2}
$$

The near-zero points dominate this sum too. The solver will sacrifice the fit everywhere else to achieve $\chi^2_{\mathrm{red}} \approx 1$ at those points. Visually, the fitted curve "nails" the minima and completely misses the broad features of the data. The $\chi^2_{\mathrm{red}}$ may look fine on paper while the plot looks terrible.

---

## 7. Fixes and Mitigations

### 7.1 Noise floor / background

Real data never reaches $I = 0$ because detector dark current, buffer subtraction errors, and photon shot noise provide a floor $I_{\min} > 0$. Capping the minimum sigma:

$$
\sigma_i \leftarrow \max\!\left(\sigma_i,\; \alpha \cdot I_{\max}\right), \quad \alpha \sim 10^{-3}\text{–}10^{-2}
$$

keeps weights finite. This is the standard practice in software like GNOM and BIFT.

### 7.2 Relative + absolute error model (Poisson-like)

$$
\sigma_i = \sqrt{\left(\frac{I_i}{k}\right)^2 + \sigma_{\min}^2}
$$

This interpolates between proportional noise at high $I$ and a constant floor at low $I$.

### 7.3 q-range truncation

Simply excluding $q$ values near the sphere minima removes the problem entirely. For real protein/polymer data this is rarely needed — real scattering has no exact zeros.

### 7.4 Adaptive $\lambda$ (L-curve / GCV)

Generalised cross-validation or the L-curve method selects $\lambda$ automatically. However, when the weight matrix is ill-conditioned due to near-zero $\sigma$ values, these methods also break down — the optimum $\lambda$ diverges or becomes numerically unstable.

### 7.5 Choose a better form factor for testing

See Section 8.

---

## 8. The Debye (Gaussian Chain) Fix

The Debye form factor for a polymer with radius of gyration $R_g$ is

$$
I_{\text{Debye}}(q) = \frac{2\left(e^{-x} - 1 + x\right)}{x^2}, \qquad x = (q R_g)^2
$$

Key properties:
- $I(0) = 1$, $I(\infty) = 0$, and $I$ is **strictly monotonically decreasing** — no zeros.
- $\sigma(q) = I(q)/k$ is therefore bounded away from zero for all finite $q$.
- $W_{ii} = k^2/I(q_i)^2$ grows only modestly as $q$ increases, so $\mathbf{A}$ remains well-conditioned.
- The $P(r)$ for a Gaussian chain is a Gaussian:

$$
P(r) \propto r^2 \exp\!\left(-\frac{3r^2}{4R_g^2}\right)
$$

which is smooth, unimodal, and easy to reconstruct — a clean benchmark.

---

## 9. Condition Number Summary

| Fixture | $I_{\min}$ | $\sigma_{\min}$ | $W_{\max}$ | $\kappa(\mathbf{A})$ | Regularisation effective? |
|---|---|---|---|---|---|
| Sphere, noiseless | $0$ (exact) | $\epsilon_{\text{dummy}}$ | moderate | moderate | ✓ yes |
| Sphere, $k=5$ | $\approx 0$ at zeros | $\approx 0$ | $\to\infty$ | $\to\infty$ | ✗ no |
| Debye, $k=5$ | $> 0$ everywhere | $> 0$ | bounded | moderate | ✓ yes |

---

## 10. Lessons

1. **The noise model must have a floor.** A purely proportional $\sigma = I/k$ is only safe when $I > 0$ everywhere in the measured range.

2. **Regularisation competes with the data weight, not the data itself.** If any weight $W_{ii}$ diverges, so does the dominant eigenvalue of $\mathbf{A}$, and $\lambda$ must diverge with it to matter — an impossible requirement.

3. **$\chi^2_{\mathrm{red}} \approx 1$ is necessary but not sufficient.** Always inspect the overlay of the fitted $I(q)$ against the data visually.

4. **Real data is kinder than synthetic data without a floor.** Bench-marking on sphere data with a pure proportional noise model is actually *harder* than real experimental conditions.

5. **The ill-posedness of the IFT problem is separate from the ill-conditioning of $\mathbf{A}$.** Regularisation addresses the former; a sensible noise model is needed to prevent the latter.

---

*Written as a companion to the UnFourier development log, March 2026.*
