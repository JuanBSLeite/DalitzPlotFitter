# Dynamics module structure

One-dimensional resonance lineshapes are organized under `src/dalitzplotfitter/dynamics/lineshape/`, with one physical model per file:

```text
dynamics/
  angular.py
  context.py
  resonance.py
  qmi2d.py
  lineshape/
    __init__.py
    common.py
    relativistic_breit_wigner.py
    gounaris_sakurai.py
    flatte.py
    pole.py
    lass.py
    kmatrix.py
    qmi.py
```

`common.py` contains only shared kinematic and Blatt-Weisskopf helpers. Public imports remain available from `dalitzplotfitter` and `dalitzplotfitter.dynamics`, so user code does not need to import implementation files directly.

Angular factors are interchangeable plugins passed through `Resonance(..., angular=...)`. The default remains `CovariantAngular()`. Two Laura++ Zemach conventions are also available:

```python
from dalitzplotfitter import Zemach_P, Zemach_Pstar

rho = Resonance(
    "rho770",
    (0, 1),
    coefficient,
    mass=0.77526,
    width=0.1491,
    spin=1,
    angular=Zemach_P(),
)
```

`Zemach_P` uses the bachelor momentum `p` evaluated in the resonance rest frame, whereas `Zemach_Pstar` uses `p*` evaluated in the parent rest frame. For spin `L`, both multiply the Laura++ phase-convention Legendre polynomial by `(p q)^L` or `(p* q)^L`, respectively. The implementation currently follows the package-wide supported angular range `L=0..4`.

These angular choices do not alter the Blatt-Weisskopf convention. The parent and resonance barrier factors remain controlled by the resonance machinery independently of the selected spin factor.

`QMI2D` remains at the `dynamics` level because it is a full two-dimensional Dalitz amplitude evaluated through `DalitzAmplitude`, rather than a one-dimensional `lineshape(mass, context)` plugin used by `Resonance`.
