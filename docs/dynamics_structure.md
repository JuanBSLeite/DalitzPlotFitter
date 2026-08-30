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

`QMI2D` remains at the `dynamics` level because it is a full two-dimensional Dalitz amplitude evaluated through `DalitzAmplitude`, rather than a one-dimensional `lineshape(mass, context)` plugin used by `Resonance`.
