"""Optional SymPy expressions compiled once into JAX lineshape kernels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number

import jax.numpy as jnp

from ...fit.parameters import Parameter, ParameterKind
from ..context import resolve_value

_CONTEXT_FIELDS = frozenset(
    {
        "parent_mass",
        "bachelor_mass",
        "spin",
        "pole_mass",
        "pole_width",
        "resonance_radius",
        "parent_radius",
        "daughter_masses[0]",
        "daughter_masses[1]",
    }
)


def _sympy():
    # Keep importing the public API independent of the optional dependency.
    try:
        import sympy
    except ImportError as exc:
        raise ImportError(
            'SympyLineshape requires SymPy; install "dalitzplotfitter[sympy]".'
        ) from exc
    return sympy


def _operations(sp):
    return {
        op.__name__: op
        for op in (
            sp.Add,
            sp.Mul,
            sp.Pow,
            sp.exp,
            sp.log,
            sp.sin,
            sp.cos,
            sp.tan,
            sp.asin,
            sp.acos,
            sp.atan,
            sp.sinh,
            sp.cosh,
            sp.tanh,
            sp.Abs,
            sp.re,
            sp.im,
            sp.conjugate,
        )
    }


def _validate_expression(sp, expression):
    if not isinstance(expression, sp.Expr):
        raise TypeError("expression must be a scalar SymPy Expr, not text or a matrix")
    operations = set(_operations(sp).values())
    for node in sp.preorder_traversal(expression):
        if type(node) is sp.Symbol or isinstance(
            node, (sp.Integer, sp.Rational, sp.Float)
        ):
            continue
        if node in (sp.I, sp.pi, sp.E):
            continue
        if node.func not in operations:
            raise ValueError(f"Unsupported symbolic operation: {node.func.__name__}")


def _context_value(context, name):
    if name == "daughter_masses[0]":
        return context.daughter_masses[0]
    if name == "daughter_masses[1]":
        return context.daughter_masses[1]
    return getattr(context, name)


@dataclass(frozen=True)
class _ResolvedSympyLineshape:
    kernel: object
    values: tuple
    context_fields: tuple[str, ...]
    complex_domain: bool

    def __call__(self, mass, context):
        m = jnp.asarray(mass)
        context = context.resolve(None)
        arguments = (
            m,
            *self.values,
            *(_context_value(context, field) for field in self.context_fields),
        )
        if self.complex_domain:
            arguments = tuple(jnp.asarray(x, dtype=complex) for x in arguments)
        result = jnp.asarray(self.kernel(*arguments), dtype=complex)
        return jnp.broadcast_to(result, m.shape)


@dataclass(frozen=True, init=False)
class SympyLineshape:
    """Optional scalar expression implementing ``lineshape(mass, context)``.

    ``parameters`` maps actual SymPy Symbols to numerical constants or existing
    ``Parameter.dynamics`` declarations. ``context_symbols`` maps Symbols to
    ResonanceContext field names (including ``daughter_masses[0]`` and ``[1]``).
    Every free symbol must have exactly one binding. ``mass_symbol`` may be
    absent from a constant expression; other unused bindings are rejected.

    The expression is compiled once with SymPy's JAX backend. Resolution keeps
    the kernel and only replaces numerical arguments, preserving autodiff and
    the existing amplitude cache's parameter discovery. No SymPy operation runs
    inside the numerical kernel. Parameters remain real scalar fit variables.

    By default, roots/logarithms have JAX's real-input domain. Set
    ``complex_domain=True`` to evaluate arguments as complex numbers, using
    principal complex branches (this does not select a physics Riemann sheet).
    Piecewise, matrices, unevaluated derivatives/integrals and arbitrary symbolic
    functions are intentionally unsupported. Inputs are trusted Python SymPy
    expressions; strings are never parsed or evaluated.

    Masses/widths are in GeV, with any additional units supplied consistently by
    the user. This expression supplies only the lineshape, not the external
    angular/barrier factors supplied by ResonanceAmplitude.
    """

    expression: object
    mass_symbol: object
    _bindings: tuple
    _context_bindings: tuple
    complex_domain: bool
    _kernel: object

    def __init__(
        self,
        expression,
        *,
        mass_symbol,
        parameters: Mapping | None = None,
        context_symbols: Mapping | None = None,
        complex_domain: bool = False,
    ):
        sp = _sympy()
        _validate_expression(sp, expression)
        parameters = dict(parameters or {})
        context_symbols = dict(context_symbols or {})
        symbols = (mass_symbol, *parameters, *context_symbols)
        if any(type(symbol) is not sp.Symbol for symbol in symbols):
            raise TypeError("Bindings and mass_symbol must use SymPy Symbol objects")
        if len(set(symbols)) != len(symbols):
            raise ValueError("Each symbol must have exactly one binding")
        if len({s.name for s in symbols}) != len(symbols):
            raise ValueError("Symbol names must be unique, including their assumptions")
        missing = expression.free_symbols - set(symbols)
        unused = (set(parameters) | set(context_symbols)) - expression.free_symbols
        if missing or unused:
            raise ValueError(
                f"Symbol bindings mismatch: missing={sorted(map(str, missing))}, "
                f"unused={sorted(map(str, unused))}"
            )
        if not isinstance(complex_domain, bool):
            raise TypeError("complex_domain must be a boolean")
        for value in parameters.values():
            if isinstance(value, Parameter):
                if value.kind is not ParameterKind.DYNAMICS or not value.owner:
                    raise ValueError(
                        "Symbolic fit parameters must be DYNAMICS with an owner"
                    )
            elif not isinstance(value, Number):
                raise TypeError(
                    "Parameter bindings must be numbers or dynamics Parameters"
                )
        for name in context_symbols.values():
            if not isinstance(name, str) or name not in _CONTEXT_FIELDS:
                raise ValueError(f"Unsupported ResonanceContext field: {name!r}")
        bindings = tuple(sorted(parameters.items(), key=lambda item: item[0].name))
        contexts = tuple(sorted(context_symbols.items(), key=lambda item: item[0].name))
        ordered = (mass_symbol, *(s for s, _ in bindings), *(s for s, _ in contexts))
        kernel = sp.lambdify(
            ordered,
            expression,
            modules="jax",
            cse=True,
            docstring_limit=0,
            use_imps=False,
            dummify=True,
        )
        for key, value in (
            ("expression", expression),
            ("mass_symbol", mass_symbol),
            ("_bindings", bindings),
            ("_context_bindings", contexts),
            ("complex_domain", complex_domain),
            ("_kernel", kernel),
        ):
            object.__setattr__(self, key, value)

    @property
    def parameters(self) -> dict:
        """A copy of the explicit bindings, understood by model discovery."""
        return dict(self._bindings)

    @property
    def context_symbols(self) -> dict:
        return dict(self._context_bindings)

    def resolve(self, values=None):
        """Bind current fit values without rebuilding or modifying the kernel."""
        return _ResolvedSympyLineshape(
            self._kernel,
            tuple(resolve_value(value, values) for _, value in self._bindings),
            tuple(name for _, name in self._context_bindings),
            self.complex_domain,
        )

    def __call__(self, mass, context):
        return self.resolve(None)(mass, context)

    def to_spec(self) -> dict:
        """Return a JSON-compatible, versioned expression/binding specification.

        Kernels, prepared samples and JAX executables are never serialized.
        ``from_spec`` accepts a parameter registry to reuse shared declarations.
        """
        from dataclasses import asdict

        sp = _sympy()
        symbols = (
            self.mass_symbol,
            *(s for s, _ in self._bindings),
            *(s for s, _ in self._context_bindings),
        )

        def encode(node):
            if type(node) is sp.Symbol:
                return {"symbol": node.name}
            if node == sp.I:
                return {"constant": "I"}
            if node == sp.pi:
                return {"constant": "pi"}
            if node == sp.E:
                return {"constant": "E"}
            if isinstance(node, sp.Rational):
                return {"rational": [int(node.p), int(node.q)]}
            if isinstance(node, sp.Float):
                return {"float": str(node), "precision": node._prec}
            return {"op": node.func.__name__, "args": [encode(a) for a in node.args]}

        def binding(value):
            if isinstance(value, Parameter):
                result = asdict(value)
                result["kind"] = value.kind.value
                return {"parameter": result}
            number = complex(value)
            return {"number": [number.real, number.imag]}

        return {
            "type": "SympyLineshape",
            "version": 1,
            "symbols": [
                {"name": s.name, "assumptions": s.assumptions0} for s in symbols
            ],
            "expression": encode(self.expression),
            "mass_symbol": self.mass_symbol.name,
            "parameters": {s.name: binding(v) for s, v in self._bindings},
            "context_symbols": {s.name: v for s, v in self._context_bindings},
            "complex_domain": self.complex_domain,
        }

    @classmethod
    def from_spec(cls, spec: Mapping, *, parameter_registry: Mapping | None = None):
        """Reconstruct allowed expression nodes, without eval or text parsing.

        An optional ``{parameter.name: Parameter}`` registry preserves identity
        with declarations also used by Resonance. Conflicting definitions fail.
        This is a lineshape specification, not full DecayModel serialization.
        """
        sp = _sympy()
        if spec.get("type") != "SympyLineshape" or spec.get("version") != 1:
            raise ValueError("Unsupported SympyLineshape specification version/type")
        symbols = {}
        for record in spec["symbols"]:
            name = record["name"]
            if name in symbols:
                raise ValueError(f"Duplicate serialized symbol: {name}")
            symbols[name] = sp.Symbol(name, **record["assumptions"])
        operations = _operations(sp)

        def decode(node):
            if set(node) == {"symbol"}:
                return symbols[node["symbol"]]
            if set(node) == {"constant"}:
                return {"I": sp.I, "pi": sp.pi, "E": sp.E}[node["constant"]]
            if set(node) == {"rational"}:
                return sp.Rational(*node["rational"])
            if set(node) == {"float", "precision"}:
                return sp.Float(node["float"], precision=node["precision"])
            if set(node) == {"op", "args"} and node["op"] in operations:
                return operations[node["op"]](*(decode(a) for a in node["args"]))
            raise ValueError("Unsupported serialized expression node")

        registry = {} if parameter_registry is None else dict(parameter_registry)
        bindings = {}
        for name, entry in spec["parameters"].items():
            if set(entry) == {"parameter"}:
                fields = dict(entry["parameter"])
                fields["kind"] = ParameterKind(fields["kind"])
                if fields["bounds"] is not None:
                    fields["bounds"] = tuple(fields["bounds"])
                value = Parameter(**fields)
                if value.name in registry:
                    if registry[value.name] != value:
                        raise ValueError(f"Conflicting parameter: {value.name}")
                    value = registry[value.name]
                else:
                    registry[value.name] = value
            elif set(entry) == {"number"}:
                real, imag = entry["number"]
                value = complex(real, imag) if imag else real
            else:
                raise ValueError("Unsupported serialized parameter binding")
            bindings[symbols[name]] = value
        return cls(
            decode(spec["expression"]),
            mass_symbol=symbols[spec["mass_symbol"]],
            parameters=bindings,
            context_symbols={symbols[s]: v for s, v in spec["context_symbols"].items()},
            complex_domain=spec["complex_domain"],
        )


__all__ = ["SympyLineshape"]
