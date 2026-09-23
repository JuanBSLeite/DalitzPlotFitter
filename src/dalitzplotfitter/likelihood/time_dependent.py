"""Tagged neutral-meson time-dependent Dalitz likelihoods (pure JAX)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from ..amplitude import PreparedAmplitudeCache


def _resolve(value, parameters):
    return value.resolve(parameters) if hasattr(value, "resolve") else value


@dataclass(frozen=True)
class NeutralMesonMixing:
    """Mixing kernel in the convention of Belle arXiv:1404.2412 Eqs. (1–2).

    x and y are fractions, NOT percentages. Time and tau use the same units;
    phi is arg(q/p) in radians. We explicitly implement the printed rate's
    +Re(C)*sinh(y*t/tau) - Im(C)*sin(x*t/tau) interference convention.
    Changing the sign convention for g_minus requires transforming amplitudes
    and/or mixing parameters consistently. All fields accept Parameter objects.
    """

    x: object = 0.0
    y: object = 0.0
    tau: object = 0.4103
    q_over_p: object = 1.0
    phi: object = 0.0

    @property
    def parameters(self):
        return tuple(
            v
            for v in (self.x, self.y, self.tau, self.q_over_p, self.phi)
            if hasattr(v, "resolve")
        )

    def resolved(self, parameters):
        return tuple(
            jnp.asarray(_resolve(v, parameters))
            for v in (self.x, self.y, self.tau, self.q_over_p, self.phi)
        )

    def valid(self, parameters):
        x, y, tau, ratio, phi = self.resolved(parameters)
        return (
            jnp.all(jnp.isfinite(jnp.stack((x, y, tau, ratio, phi))))
            & (jnp.abs(y) < 1)
            & (tau > 0)
            & (ratio > 0)
        )

    def basis(self, time, parameters):
        """Return |g+|², |g-|², Re(g+* g-), Im(g+* g-)."""
        x, y, tau, _, _ = self.resolved(parameters)
        u = jnp.maximum(jnp.asarray(time), 0) / tau
        e1 = jnp.exp((-1 + y + 1j * x) * u / 2)
        e2 = jnp.exp((-1 - y - 1j * x) * u / 2)
        gp, gm = (e1 + e2) / 2, (e1 - e2) / 2
        cross = jnp.conj(gp) * gm
        result = jnp.stack(
            (jnp.abs(gp) ** 2, jnp.abs(gm) ** 2, jnp.real(cross), jnp.imag(cross)),
            axis=-1,
        )
        return jnp.where((jnp.asarray(time) >= 0)[..., None], result, 0)

    def integrals(self, time_range, parameters):
        """Exact time integrals for unit acceptance and perfect resolution."""
        x, y, tau, _, _ = self.resolved(parameters)
        low, high = time_range
        low = max(low, 0.0)

        def integrate(rate):
            start = jnp.exp(-rate * low / tau)
            if np.isinf(high):
                return tau * start / rate
            return tau * start * (-jnp.expm1(-rate * (high - low) / tau)) / rate

        a, b, c = integrate(1 - y), integrate(1 + y), integrate(1 - 1j * x)
        return jnp.stack(
            (
                (a + b + 2 * jnp.real(c)) / 4,
                (a + b - 2 * jnp.real(c)) / 4,
                (a - b) / 4,
                jnp.imag(c) / 2,
            )
        )


def _rate(basis, aa, bb, cross):
    return (
        basis[..., 0] * aa
        + basis[..., 1] * bb
        + 2 * (basis[..., 2] * jnp.real(cross) - basis[..., 3] * jnp.imag(cross))
    )


@dataclass(frozen=True)
class TimeDependentDalitzNLL:
    """Signal NLL conditional on observed initial-flavour tag and sigma_t.

    A single cache contains A components first and Abar components last, all
    evaluated at the SAME final-state coordinates and on the SAME integration
    sample. ``n_particle_components`` gives the split. Component names must be
    unique. The cache's efficiency_normalization includes Dalitz acceptance
    and veto; ``efficiency`` supplies matching data-side factors.

    tags: +1 = D0, -1 = D0bar. wrong_tag is the posterior probability that the
    observed tag is wrong IN THE SELECTED SAMPLE; normalized flavour PDFs are
    mixed, not raw rates. It may be scalar or event-wise, fixed or Parameter.

    Default: perfect time resolution, unit temporal acceptance, t in [0, inf).
    For acceptance or Gaussian sigma_t, supply positive true-time quadrature
    nodes/weights approximating integral dt as SUM(weights*f), not mean.
    time_acceptance(t, parameters) acts on TRUE time and must factorize from
    Dalitz acceptance. sigma_t is fixed, positive, scalar or event-wise.
    Gaussian normalization includes the observed time_range (negative measured
    times allowed). Quadrature truncation/convergence is the caller's duty.

    This is not an extended likelihood or a production/initial-tag asymmetry
    measurement. Supply all cache and mixing parameters to Minimizer explicitly;
    call cache.check_parameters before fitting separately constructed lists.
    """

    cache: PreparedAmplitudeCache
    n_particle_components: int
    times: object
    tags: object
    mixing: NeutralMesonMixing
    efficiency: object = 1.0
    wrong_tag: object = 0.0
    time_range: tuple = (0.0, np.inf)
    time_nodes: object = None
    time_weights: object = None
    time_acceptance: object = None
    sigma_t: object = None

    def __post_init__(self):
        n = self.cache.data_components.shape[0]
        times, tags = np.asarray(self.times), np.asarray(self.tags)
        if times.shape != (n,) or tags.shape != (n,) or n == 0:
            raise ValueError("times and tags must match the non-empty cache data")
        if not np.all(np.isfinite(times)) or not np.all(np.isin(tags, [-1, 1])):
            raise ValueError("times must be finite; tags must be +1 or -1")
        if not 0 < self.n_particle_components < len(self.cache.components):
            raise ValueError("both flavour component groups must be non-empty")
        low, high = self.time_range
        if not np.isfinite(low) or not low < high or high <= 0:
            raise ValueError("invalid observed time_range")
        if np.any((times < low) | (times > high)):
            raise ValueError("data outside observed time_range")
        if self.sigma_t is None and (low < 0 or np.any(times < 0)):
            raise ValueError("negative times require a resolution model")
        eff = np.asarray(self.efficiency)
        if eff.shape not in ((), (n,)) or not np.all(np.isfinite(eff) & (eff >= 0)):
            raise ValueError("efficiency must be nonnegative scalar or event array")
        if self.sigma_t is not None:
            sigma = np.asarray(self.sigma_t)
            if sigma.shape not in ((), (n,)) or not np.all(
                np.isfinite(sigma) & (sigma > 0)
            ):
                raise ValueError("sigma_t must be positive scalar or event array")
        quadrature = self.time_nodes is not None
        if quadrature != (self.time_weights is not None):
            raise ValueError("time_nodes and time_weights must be supplied together")
        if (
            self.time_acceptance is not None or self.sigma_t is not None
        ) and not quadrature:
            raise ValueError("acceptance/resolution requires true-time quadrature")
        if quadrature:
            t, w = np.asarray(self.time_nodes), np.asarray(self.time_weights)
            if (
                t.ndim != 1
                or t.size == 0
                or t.shape != w.shape
                or not np.all(np.isfinite(t) & (t >= 0))
                or not np.all(np.isfinite(w) & (w > 0))
            ):
                raise ValueError("invalid true-time quadrature")
        wt = np.asarray(_resolve(self.wrong_tag, {}))
        if wt.shape not in ((), (n,)) or not np.all(
            np.isfinite(wt) & (wt >= 0) & (wt <= 1)
        ):
            raise ValueError("wrong_tag must be a probability scalar or event array")

    def _time_basis(self, parameters, times=None):
        times = jnp.asarray(self.times if times is None else times)
        basis = self.mixing.basis(times, parameters)
        if self.time_nodes is None:
            return basis, self.mixing.integrals(self.time_range, parameters), True
        nodes, weights = jnp.asarray(self.time_nodes), jnp.asarray(self.time_weights)
        acceptance = (
            jnp.ones_like(nodes)
            if self.time_acceptance is None
            else jnp.broadcast_to(self.time_acceptance(nodes, parameters), nodes.shape)
        )
        valid = jnp.all(jnp.isfinite(acceptance) & (acceptance >= 0))
        weighted = (
            self.mixing.basis(nodes, parameters) * (weights * acceptance)[:, None]
        )
        low, high = self.time_range
        if self.sigma_t is None:
            selected = (nodes >= low) & (nodes <= high)
            integrals = jnp.sum(weighted * selected[:, None], axis=0)
            if self.time_acceptance is not None:
                acc_data = jnp.broadcast_to(
                    self.time_acceptance(times, parameters), times.shape
                )
                valid &= jnp.all(jnp.isfinite(acc_data) & (acc_data >= 0))
                basis = basis * acc_data[:, None]
            return basis, integrals, valid
        sigma = jnp.broadcast_to(jnp.asarray(self.sigma_t), times.shape)

        def accumulate(carry, item):
            node, kernel = item
            gaussian = jnp.exp(-0.5 * ((times - node) / sigma) ** 2) / (
                jnp.sqrt(2 * jnp.pi) * sigma
            )
            probability = ndtr((high - node) / sigma) - ndtr((low - node) / sigma)
            return (
                carry[0] + gaussian[:, None] * kernel,
                carry[1] + probability[:, None] * kernel,
            ), None

        (basis, integrals), _ = jax.lax.scan(
            accumulate,
            (jnp.zeros_like(basis), jnp.zeros_like(basis)),
            (nodes, weighted),
        )
        return basis, integrals, valid

    def _overlap_and_ratio(self, parameters):
        """Dalitz-integrated (ia, ib, cross) overlap plus the resolved q/p ratio.

        Shared by ``densities()`` (per-event ``amplitudes`` too) and
        ``dalitz_integrated_time_pdf()`` (Dalitz-integrated curves only).
        """
        groups = jnp.arange(len(self.cache.components)) < self.n_particle_components
        groups = jnp.stack((groups, ~groups), axis=1)
        amplitudes, overlap = self.cache.coherent_groups(parameters, groups)
        ia, ib = jnp.real(overlap[0, 0]), jnp.real(overlap[1, 1])
        cross = overlap[0, 1]
        _, _, _, magnitude, phase = self.mixing.resolved(parameters)
        ratio = magnitude * jnp.exp(1j * phase)
        return amplitudes, ia, ib, cross, ratio

    def densities(self, parameters):
        """Joint (Dalitz,time) densities conditional on observed tag, sigma_t."""
        amplitudes, ia, ib, cross, ratio = self._overlap_and_ratio(parameters)
        a, b = amplitudes[:, 0], amplitudes[:, 1]
        basis, integral, valid = self._time_basis(parameters)
        plus = _rate(
            basis, jnp.abs(a) ** 2, jnp.abs(ratio * b) ** 2, ratio * jnp.conj(a) * b
        )
        minus = _rate(
            basis, jnp.abs(b) ** 2, jnp.abs(a / ratio) ** 2, jnp.conj(b) * a / ratio
        )
        norm_plus = _rate(integral, ia, jnp.abs(ratio) ** 2 * ib, ratio * cross)
        norm_minus = _rate(
            integral, ib, ia / jnp.abs(ratio) ** 2, jnp.conj(cross) / ratio
        )
        p, m = plus / norm_plus, minus / norm_minus
        wrong = jnp.asarray(_resolve(self.wrong_tag, parameters))
        pdf = jnp.where(
            jnp.asarray(self.tags) == 1,
            (1 - wrong) * p + wrong * m,
            (1 - wrong) * m + wrong * p,
        )
        valid &= (
            self.mixing.valid(parameters)
            & jnp.all(jnp.isfinite(wrong) & (wrong >= 0) & (wrong <= 1))
            & jnp.all(norm_plus > 0)
            & jnp.all(norm_minus > 0)
        )
        return jnp.where(valid, jnp.asarray(self.efficiency) * pdf, jnp.nan)

    def dalitz_integrated_time_pdf(self, times, parameters):
        """Exact, Dalitz-marginalized decay-time density for each observed tag.

        Returns ``(observed_plus, observed_minus)``: each is a density over
        ``self.time_range`` (integrates to 1 there) for the tag=+1 and tag=-1
        populations, already including wrong_tag mixing -- the same physical
        curve ``densities()`` normalizes each event against, evaluated at
        ``times`` instead of at the fixed events in ``self.times``. Intended
        for plotting/diagnostics (see ``TimeDependentFitSession.plot_time_projection``),
        not for use inside a fit objective.

        Requires unit temporal acceptance and perfect time resolution
        (``self.time_nodes is None``): this helper currently implements only
        the analytic time basis.
        Factorized acceptance and a scalar resolution could also be
        marginalized, but are not implemented in this helper. Also
        requires a scalar ``wrong_tag``: a single curve needs one
        representative mistag probability, not the per-event values a fit may
        use.
        """
        if self.time_nodes is not None:
            raise ValueError(
                "dalitz_integrated_time_pdf requires unit temporal acceptance "
                "and perfect resolution (time_nodes=None)"
            )
        wrong = jnp.asarray(_resolve(self.wrong_tag, parameters))
        if wrong.ndim != 0:
            raise ValueError(
                "dalitz_integrated_time_pdf requires a scalar wrong_tag; a "
                "single curve needs one representative mistag probability"
            )
        _, ia, ib, cross, ratio = self._overlap_and_ratio(parameters)
        basis = self.mixing.basis(jnp.asarray(times), parameters)
        integral = self.mixing.integrals(self.time_range, parameters)
        r2 = jnp.abs(ratio) ** 2
        rate_plus = _rate(basis, ia, r2 * ib, ratio * cross)
        rate_minus = _rate(basis, ib, ia / r2, jnp.conj(cross) / ratio)
        norm_plus = _rate(integral, ia, r2 * ib, ratio * cross)
        norm_minus = _rate(integral, ib, ia / r2, jnp.conj(cross) / ratio)
        p_plus, p_minus = rate_plus / norm_plus, rate_minus / norm_minus
        observed_plus = (1 - wrong) * p_plus + wrong * p_minus
        observed_minus = (1 - wrong) * p_minus + wrong * p_plus
        return observed_plus, observed_minus

    def _diagnostic_time_basis(self, times, parameters):
        """Reuse the fitted response for diagnostics with scalar conditioning."""
        wrong = jnp.asarray(_resolve(self.wrong_tag, parameters))
        if wrong.ndim != 0:
            raise ValueError("Dalitz diagnostics require a scalar wrong_tag")
        if self.sigma_t is not None and jnp.ndim(self.sigma_t) != 0:
            raise ValueError("Dalitz diagnostics require a scalar sigma_t")
        basis, integral, valid = self._time_basis(
            parameters,
            jnp.atleast_1d(jnp.asarray(times)),
        )
        # A scalar resolution broadcasts the normalization over the requested
        # times; all rows have the same selected-time integral.
        if integral.ndim == 2:
            integral = integral[0]
        return basis, integral, wrong, valid

    def _diagnostic_efficiency(self, efficiency):
        if efficiency is None:
            if self.cache.efficiency_normalization is not None:
                raise ValueError(
                    "supply efficiency at the requested Dalitz points to match "
                    "the acceptance-weighted cache"
                )
            return 1.0
        return jnp.asarray(efficiency)

    def tag_marginal_density(
        self,
        amplitude_a,
        amplitude_b,
        parameters,
        *,
        efficiency=None,
    ):
        """Selected, time-integrated Dalitz densities for the two observed tags.

        Amplitudes at arbitrary points must use the fitted coefficients,
        dynamics and component scales of this cache. Supply ``efficiency``
        (including vetoes) at those points when the cache includes acceptance.
        Each returned density integrates to one under ``mean(weights*f)``.

        Uses the same selected-time integrals as ``densities()``, including
        true-time acceptance and Gaussian resolution. Requires scalar
        ``wrong_tag`` and scalar ``sigma_t`` (if present): an event-wise
        response needs an explicitly specified conditioning distribution.
        """
        _, integral, wrong, valid = self._diagnostic_time_basis(
            jnp.asarray(self.times)[:1],
            parameters,
        )
        eff = self._diagnostic_efficiency(efficiency)
        _, ia, ib, cross, ratio = self._overlap_and_ratio(parameters)
        r2 = jnp.abs(ratio) ** 2
        a, b = jnp.asarray(amplitude_a), jnp.asarray(amplitude_b)
        rate_plus = _rate(
            integral, jnp.abs(a) ** 2, jnp.abs(ratio * b) ** 2, ratio * jnp.conj(a) * b
        )
        rate_minus = _rate(
            integral, jnp.abs(b) ** 2, jnp.abs(a) ** 2 / r2, jnp.conj(b) * a / ratio
        )
        norm_plus = _rate(integral, ia, r2 * ib, ratio * cross)
        norm_minus = _rate(integral, ib, ia / r2, jnp.conj(cross) / ratio)
        p_plus, p_minus = rate_plus / norm_plus, rate_minus / norm_minus
        observed_plus = eff * ((1 - wrong) * p_plus + wrong * p_minus)
        observed_minus = eff * ((1 - wrong) * p_minus + wrong * p_plus)
        return (
            jnp.where(valid, observed_plus, jnp.nan),
            jnp.where(valid, observed_minus, jnp.nan),
        )

    def dalitz_density_at_time(
        self,
        amplitude_a,
        amplitude_b,
        t,
        parameters,
        *,
        efficiency=None,
    ):
        """Dalitz densities conditional on observed tag and observed time ``t``.

        First mix the selected-sample joint flavour PDFs with ``wrong_tag``,
        then divide by their observed-tag time marginal. Thus the posterior
        flavour composition at ``t`` can differ from the sample-wide mistag.
        For zero mistag and perfect resolution, t=0 reduces to |A|²/I_A or
        |Abar|²/I_Abar, multiplied by the Dalitz acceptance.

        Amplitude scales, efficiency, scalar wrong_tag/sigma_t requirements
        are as in ``tag_marginal_density``. Acceptance and Gaussian smearing
        use the same true-time quadrature as the likelihood. An out-of-range
        time or a zero-probability time has no conditional density (NaN).
        """
        if jnp.ndim(t) != 0:
            raise ValueError("dalitz_density_at_time requires a scalar time")
        basis, integral, wrong, valid = self._diagnostic_time_basis(t, parameters)
        basis = basis[0]
        eff = self._diagnostic_efficiency(efficiency)
        _, ia, ib, cross, ratio = self._overlap_and_ratio(parameters)
        r2 = jnp.abs(ratio) ** 2
        a, b = jnp.asarray(amplitude_a), jnp.asarray(amplitude_b)
        rate_plus = _rate(
            basis, jnp.abs(a) ** 2, jnp.abs(ratio * b) ** 2, ratio * jnp.conj(a) * b
        )
        rate_minus = _rate(
            basis, jnp.abs(b) ** 2, jnp.abs(a) ** 2 / r2, jnp.conj(b) * a / ratio
        )
        norm_plus = _rate(integral, ia, r2 * ib, ratio * cross)
        norm_minus = _rate(integral, ib, ia / r2, jnp.conj(cross) / ratio)
        time_plus = _rate(basis, ia, r2 * ib, ratio * cross) / norm_plus
        time_minus = _rate(basis, ib, ia / r2, jnp.conj(cross) / ratio) / norm_minus
        p_plus, p_minus = rate_plus / norm_plus, rate_minus / norm_minus
        observed_plus = eff * ((1 - wrong) * p_plus + wrong * p_minus)
        observed_minus = eff * ((1 - wrong) * p_minus + wrong * p_plus)
        observed_plus /= (1 - wrong) * time_plus + wrong * time_minus
        observed_minus /= (1 - wrong) * time_minus + wrong * time_plus
        valid = valid & (t >= self.time_range[0]) & (t <= self.time_range[1])
        return (
            jnp.where(valid, observed_plus, jnp.nan),
            jnp.where(valid, observed_minus, jnp.nan),
        )

    def _physical_parameters(self, parameters):
        """Cheap, data-independent gate for __call__.

        mixing.basis()/integrals() divide raw by tau and by (1-y), (1+y),
        (1-i*x) with no internal clamp; jnp.where only masks densities()'s
        *output*, so both its branches are still traced and differentiated.
        Checking validity here and gating the whole evaluation with
        jax.lax.cond (as cp.py/mixture.py do) keeps an unphysical point from
        ever tracing those divisions, so it cannot poison jax.grad with NaN.
        """
        wrong = jnp.asarray(_resolve(self.wrong_tag, parameters))
        return self.mixing.valid(parameters) & jnp.all(
            jnp.isfinite(wrong) & (wrong >= 0) & (wrong <= 1)
        )

    def __call__(self, parameters):
        """Return +inf outside the physical domain, including during JIT fits."""
        dtype = self.cache.data_components.real.dtype

        def evaluate(_):
            pdf = self.densities(parameters)
            valid = jnp.all(jnp.isfinite(pdf) & (pdf > 0))
            nll = -jnp.sum(jnp.log(jnp.where(pdf > 0, pdf, 1)))
            return jnp.asarray(jnp.where(valid, nll, jnp.inf), dtype=dtype)

        return jax.lax.cond(
            self._physical_parameters(parameters),
            evaluate,
            lambda _: jnp.asarray(jnp.inf, dtype=dtype),
            operand=None,
        )
