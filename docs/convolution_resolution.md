# PDF convolution and detector resolution

DalitzPlotFitter provides a generic one-dimensional convolution layer for observables whose reconstructed value differs continuously from the underlying true value.

For a true PDF `f(t)` and a conditional resolution kernel `R(x | t)`, the observed density is

\[
g(x)=\frac{\int f(t)R(x\mid t)\,dt}{\int_{x_\min}^{x_\max}dx\int f(t)R(x\mid t)\,dt}.
\]

The finite observed-range normalization is explicit. This matters whenever detector resolution migrates probability outside the fitted window.

## Reuse the relativistic Breit-Wigner from the amplitude model

For an isolated resonance, `LineshapeIntensity1D` converts an existing complex dynamics plugin into a normalized one-dimensional intensity PDF. No second Breit-Wigner implementation is needed.

```python
from dalitzplotfitter import (
    ConvolvedPDF1D,
    GaussianResolution1D,
    LineshapeIntensity1D,
    RelativisticBreitWigner,
    ResonanceContext,
)

context = ResonanceContext(
    parent_mass=5.27934,
    daughter_masses=(0.493677, 0.13957039),
    bachelor_mass=0.13957039,
    spin=1,
    pole_mass=0.8958,
    pole_width=0.0474,
    resonance_radius=4.0,
    parent_radius=4.0,
)

true_mass = LineshapeIntensity1D.from_context(
    RelativisticBreitWigner(),
    context,
    quadrature_order=512,
)
```

`from_context` uses the full physical pair-mass interval

\[
m_{\min}=m_1+m_2,\qquad m_{\max}=M-m_{\rm bachelor}.
\]

The PDF is

\[
f(m)=\frac{|R(m)|^2}{\int_{m_{\min}}^{m_{\max}} |R(m')|^2\,dm'},
\]

where `RelativisticBreitWigner` is the same implementation used by the Dalitz amplitude model, including the running width and Blatt-Weisskopf convention.

## Gaussian detector response

```python
resolution = GaussianResolution1D(sigma=0.008, bias=0.0)

reconstructed_mass = ConvolvedPDF1D(
    true_mass,
    resolution,
    true_low=true_mass.low,
    true_high=true_mass.high,
    observed_low=true_mass.low,
    observed_high=true_mass.high,
    quadrature_order=192,
)
```

`sigma` and `bias` may be ordinary values or fit `Parameter` objects.

For the Gaussian kernel, the convolution numerator is evaluated with Gauss--Hermite quadrature local to each reconstructed value. This remains accurate when the detector resolution is narrow compared with the full mass interval. The finite-window normalization uses Gauss--Legendre quadrature together with the analytic Gaussian interval probability.

`quadrature_order` is a numerical accuracy setting only; it is not a spin or resonance parameter.

## Coherent amplitudes and interference

`LineshapeIntensity1D` is appropriate for an isolated lineshape demonstration. If several amplitudes interfere, detector resolution must act on the complete coherent intensity

\[
|\mathcal A(m)|^2=\left|\sum_r c_r A_r(m)\right|^2,
\]

not on each component intensity independently. Convolving individual `|A_r|^2` terms would remove the interference terms.

This distinction is important for future extensions that apply detector resolution directly to fitted amplitude-model projections.

## Convolution versus Dalitz migration

`ConvolvedPDF1D` and `SquareDalitzSCFMap` solve different detector-resolution problems.

- `ConvolvedPDF1D` describes continuous smearing of a one-dimensional observable such as invariant mass, decay time, or another discriminating variable.
- `SquareDalitzSCFMap` describes migration between true and reconstructed regions of the two-dimensional Square-Dalitz plane for self-cross-feed/misreconstruction.

A genuine multidimensional Dalitz-resolution treatment should therefore use a migration kernel/operator rather than independent one-dimensional convolutions of the Dalitz invariants.

See `notebooks/20_pdf_convolution_resolution.ipynb` for the relativistic `K*(892)0 -> K pi` example.
