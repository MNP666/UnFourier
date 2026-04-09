# Fixing boundary discontinuities in P(r)

## The symptom

When plotting P(r), the curve appeared to drop discontinuously to zero at r = 0
and r = D_max.  The effect was subtle at low smoothness but became extreme when
`d1_smoothness` was increased: the solver produced a flat plateau across the
interior that fell vertically to zero at both ends — a top-hat rather than a
smooth bell-shaped distribution.

## Root cause

### The regulariser is blind to the boundary

The finite-difference regularisers (`FirstDerivative`, `SecondDerivative`,
`CombinedDerivative`) operate purely in coefficient space.  For a basis with
n free parameters they build matrices of size (n−1)×n and (n−2)×n respectively
and penalise differences *between adjacent coefficients*.

The critical gap: neither operator knows that the coefficient vector is
implicitly padded with zeros at both ends.  For the `FirstDerivative` on n
coefficients:

```
penalised differences:  c[1]−c[0],  c[2]−c[1],  …,  c[n-1]−c[n-2]
```

The jump from the boundary (P = 0) to c[0], and from c[n-1] to the boundary
(P = 0), is **not in this list**.

### Why `d1_smoothness` makes it worse

With a high first-derivative weight the solver minimises differences between
adjacent interior coefficients, driving them toward a single constant value.
But the boundary condition (P = 0 at both ends) is enforced by a separate
mechanism — either structural exclusion (spline) or soft constraint rows
(rect).  These two mechanisms are independent.

Result: the solver is free to set c[0] = c[1] = … = c[n-1] = K for some
constant K, satisfying the interior smoothness criterion perfectly, while
the separately enforced P(0) = P(D_max) = 0 creates vertical cliffs at both
edges.  The higher the smoothness weight, the flatter the plateau, and the
steeper the cliff.

### Basis-specific manifestation

**Rect basis** — Bin centres live at (j + 0.5)·Δr so the first bin is at
Δr/2, not at r = 0.  The `append_boundary_constraints` function added two
pseudo-data rows to pin c[0] → 0 and c[n-1] → 0, but this was independent of
the regulariser.  With `d1_smoothness` pulling c[0] ≈ c[1] and the boundary
constraint pulling c[0] → 0, the two forces created a cliff between c[0] and
c[1].

**Spline basis** — The two endpoint B-splines (B_0 at r = 0 and B_{n-1} at
r = D_max) were structurally excluded from the design matrix.  P(0) = P(D_max)
= 0 held for any coefficient vector.  However, the regulariser saw only the n
free coefficients; it had no knowledge of the implicit c_0 = c_{n+1} = 0
values.  Demanding coefficient smoothness again produced a plateau in the
interior that the B-spline representation then forced abruptly to zero at the
boundaries through the basis functions' compact support.

## The fix

### Include boundary coefficients in the design matrix

Rather than treating the boundary conditions as a separate step, both bases
now include explicit boundary basis functions in the design matrix.  Their
coefficients are pinned to zero by the existing `append_boundary_constraints`
mechanism, which the regulariser now also sees.

**Rect basis** (`UniformGrid`, `src/basis.rs`):

The r-grid grows from n points to n + 2:

```
before:  [Δr/2,  3Δr/2,  …,  r_max − Δr/2]          n entries
after:   [0,  Δr/2,  3Δr/2,  …,  r_max − Δr/2,  r_max]   n+2 entries
```

The two new entries at r = 0 and r = r_max are *ghost bins* with zero kernel
contribution (zero width, Δr = 0).  They do not affect I(q) at all.  Their
sole purpose is to give the regulariser concrete positions to diff against:

```
FirstDerivative row 0:   c[1] − c[0]   (c[0] = 0 by constraint → penalises c[1])
FirstDerivative row n:   c[n+1] − c[n] (c[n+1] = 0 by constraint → penalises c[n])
```

**Spline basis** (`CubicBSpline`, `src/basis.rs`):

The two endpoint B-splines B_0 and B_{n-1} are re-included in the design
matrix instead of being dropped.  `r_values()` now returns all n_free + 2
Greville abscissae (including the endpoints at 0 and r_max), and
`build_kernel_matrix()` returns the full (n_free + 2)-column matrix.

The endpoint B-splines have non-trivial kernel contributions, but since their
coefficients are pinned to zero by `append_boundary_constraints` they
contribute nothing to I_calc.  What they do provide is the regulariser anchor:
the penalty now includes c_1 − c_0 = c_1 (slope from left boundary) and
c_{n+1} − c_n = −c_n (slope to right boundary).

**`main.rs`**:

- `boundary_multiplier` defaults to `Some(1.0)` (always-on) instead of `None`.
  Both bases need boundary constraints to pin the new ghost/endpoint coefficients.
- The `if matches!(args.basis, BasisChoice::Rect)` guard on the boundary weight
  calculation is removed; the same weight formula now applies to both bases.
- The post-solve boundary point insertion for the spline (`solution.r.insert(0,
  0.0)` etc.) is removed.  The boundary r-values are now part of `r_values()`
  and appear in the output naturally.

## Effect on the regulariser

Before the fix, for n free parameters the regulariser built an (n+1)×n
first-derivative matrix:

```
⎡−1  1  0  0  …  0⎤
⎢ 0 −1  1  0  …  0⎥
⎣ 0  0  0  …  −1 1⎦
```

After the fix it builds an (n+1)×(n+2) matrix (one row per adjacent pair across
all n+2 positions, including the pinned boundaries):

```
⎡−1  1  0  0  …  0  0⎤   ← c[1] − c[0],  c[0] pinned → penalises c[1]
⎢ 0 −1  1  0  …  0  0⎥
⎣ 0  0  0  …  0 −1  1⎦   ← c[n+1] − c[n], c[n+1] pinned → penalises c[n]
```

The second-derivative matrix gains two analogous rows.  With `d1_smoothness`
set high, the solver can no longer create a flat plateau without also paying a
large penalty for the boundary slopes.  The plateau is forced to taper smoothly
to zero at both ends.

## Configuration

The `boundary_weight` field in `unfourier.toml` controls the constraint
strength:

| Value | Behaviour |
|-------|-----------|
| absent (default) | auto weight, multiplier = 1.0 |
| `0.0` | same as absent |
| `5.0` | auto weight × 5 (stronger pinning, useful for spline) |
| `-1.0` | disabled (not recommended — boundary ghosts are uncontrolled) |

For most datasets the default weight is sufficient.  If the spline boundary
values visibly deviate from zero, increase the multiplier to 5–10.
