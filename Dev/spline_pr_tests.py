import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

def make_spline(weights: np.ndarray, r_max: float) -> CubicSpline:
    """
    Build a cubic spline on [0, r_max] with compact support.
    
    The spline passes through zero at x=0 and x=r_max (clamped BCs),
    with interior knot values given by `weights`.
    
    Parameters
    ----------
    weights : array of shape (n,)
        Values at n interior knots (uniformly spaced between 0 and r_max).
    r_max : float
        Right boundary; spline is zero here.
    
    Returns
    -------
    CubicSpline
        A callable spline defined on [0, r_max].
    """
    n = len(weights)
    # Knots: 0, interior points, r_max
    x = np.linspace(0.0, r_max, n + 2)
    y = np.concatenate([[0.0], weights, [0.0]])

    # "clamped" BCs: first derivative = 0 at both ends
    # swap to "not-a-knot" or "natural" if you prefer
    cs = CubicSpline(x, y, bc_type="clamped")
    return cs

#%%
r_max = 50.0
test_cases = [[0.1,0.2,0.6,0.2,0.1],
              [0.2,1.0,0.6,0.4,0.2,0.1],
              [0.2,0.6,0.5,0.2],
              [0.1,0.3,0.2,0.5,0.2],
              [0.1,0.4,0.1],
              [0.3,0.8,0.4,0.2,0.3,0.1]]

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

for test, ax in zip(test_cases, axes.flatten()):
    cs = make_spline(test, r_max)

    ax.plot(r, cs(r))
    ax.scatter(np.linspace(0, r_max, len(test) + 2),
               np.concatenate([[0], test, [0]]),
               color="red", zorder=5, label="knots")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"weights: {test}")

#%%






r_max = 50.0
n_weights = 4
n_trials = 12
r = np.linspace(0, r_max, 500)

rng = np.random.default_rng(seed=42)

fig, axes = plt.subplots(4, 3, figsize=(12, 6), sharex=True, sharey=True)

for ax in axes.flat:
    weights = rng.uniform(0.2, 2.0, size=n_weights)
    cs = make_spline(weights, r_max)
    ax.plot(r, cs(r))
    ax.scatter(np.linspace(0, r_max, n_weights + 2),
               np.concatenate([[0], weights, [0]]),
               color="red", zorder=5, label="knots")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"weights: {weights.round(2)}")

plt.tight_layout()
plt.show()


#%%


