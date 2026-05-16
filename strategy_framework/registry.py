"""Extension registry for optimizer-discovered signal components.

New indicator or pattern modules can call ``register_component`` at import
time. The enhanced strategy builder will register those scorers on the
``ConfluenceScorer``, and the auto optimizer will sample the declared parameter
space automatically.

Example plugin::

    import numpy as np
    from strategy_framework.registry import ParamSpec, register_component

    def factory(params):
        lookback = int(params["my_lookback"])

        def scorer(bars, ctx):
            close = bars["close"].to_numpy(float)
            signal = np.zeros(close.shape[0], dtype=float)
            signal[lookback:] = close[lookback:] > close[:-lookback]
            return signal, 1.0 - signal

        return scorer

    register_component(
        "my_pattern",
        factory=factory,
        params=(ParamSpec.integer("my_lookback", 8, 80, default=20),),
        default_weight=0.0,
    )
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import optuna


ParamKind = Literal["float", "int", "categorical"]
ScorerFn = Callable[[Any, dict[str, Any]], tuple[Any, Any]]
ScorerFactory = Callable[[dict[str, Any]], ScorerFn]


@dataclass(frozen=True)
class ParamSpec:
    """One optimizer-searchable parameter for a registered component."""

    name: str
    kind: ParamKind
    low: float | int | None = None
    high: float | int | None = None
    default: float | int | str | None = None
    choices: tuple[str, ...] = ()

    @classmethod
    def floating(
        cls,
        name: str,
        low: float,
        high: float,
        *,
        default: float | None = None,
    ) -> "ParamSpec":
        return cls(name=name, kind="float", low=low, high=high, default=default)

    @classmethod
    def integer(
        cls,
        name: str,
        low: int,
        high: int,
        *,
        default: int | None = None,
    ) -> "ParamSpec":
        return cls(name=name, kind="int", low=low, high=high, default=default)

    @classmethod
    def categorical(
        cls,
        name: str,
        choices: Iterable[str],
        *,
        default: str | None = None,
    ) -> "ParamSpec":
        vals = tuple(str(v) for v in choices)
        if not vals:
            raise ValueError(f"{name}: categorical choices cannot be empty")
        return cls(
            name=name,
            kind="categorical",
            default=default if default is not None else vals[0],
            choices=vals,
        )

    def suggest(self, trial: optuna.Trial) -> Any:
        if self.kind == "float":
            return trial.suggest_float(self.name, float(self.low), float(self.high))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high))
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        raise ValueError(f"unknown ParamSpec kind: {self.kind!r}")

    def numeric_bound(self) -> tuple[float, float, bool] | None:
        if self.kind == "float":
            return float(self.low), float(self.high), False
        if self.kind == "int":
            return float(self.low), float(self.high), True
        return None


@dataclass(frozen=True)
class ComponentSpec:
    """A scorer component and its search space."""

    name: str
    factory: ScorerFactory
    params: tuple[ParamSpec, ...] = ()
    weight_min: float = 0.0
    weight_max: float = 2.5
    default_weight: float = 0.0

    @property
    def weight_name(self) -> str:
        return f"w_{self.name}"

    def make_scorer(self, sampled_params: dict[str, Any]) -> ScorerFn:
        values = {
            spec.name: sampled_params.get(spec.name, spec.default)
            for spec in self.params
        }
        return self.factory(values)


_COMPONENTS: dict[str, ComponentSpec] = {}


def register_component(
    name: str,
    *,
    scorer: ScorerFn | None = None,
    factory: ScorerFactory | None = None,
    params: Iterable[ParamSpec] = (),
    weight_min: float = 0.0,
    weight_max: float = 2.5,
    default_weight: float = 0.0,
    replace: bool = False,
) -> ComponentSpec:
    """Register a scorer component for enhanced-strategy optimization.

    Use ``scorer=`` for a scorer with no component-specific parameters, or
    ``factory=`` when parameter values must be closed over before scoring.
    """
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("component name must be alphanumeric/underscore")
    if scorer is None and factory is None:
        raise ValueError("register_component requires scorer= or factory=")
    if scorer is not None and factory is not None:
        raise ValueError("use scorer= or factory=, not both")
    if name in _COMPONENTS and not replace:
        raise ValueError(f"component already registered: {name}")

    if scorer is not None:
        factory = lambda _params, fn=scorer: fn

    spec = ComponentSpec(
        name=name,
        factory=factory,  # type: ignore[arg-type]
        params=tuple(params),
        weight_min=float(weight_min),
        weight_max=float(weight_max),
        default_weight=float(default_weight),
    )
    _COMPONENTS[name] = spec
    return spec


def registered_components() -> tuple[ComponentSpec, ...]:
    return tuple(_COMPONENTS.values())


def clear_registered_components() -> None:
    """Clear runtime-registered components. Intended for tests/smoke runs."""
    _COMPONENTS.clear()


def load_component_modules(modules: Iterable[str] | None) -> list[str]:
    """Import dotted modules or ``.py`` files that register components."""
    loaded: list[str] = []
    for raw in modules or ():
        ref = str(raw).strip()
        if not ref:
            continue
        path = Path(ref)
        if path.suffix == ".py" or path.exists():
            resolved = path.resolve()
            name = f"strategy_component_{resolved.stem}_{abs(hash(resolved))}"
            spec = importlib.util.spec_from_file_location(name, resolved)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import component module: {ref}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            loaded.append(str(resolved))
        else:
            importlib.import_module(ref)
            loaded.append(ref)
    return loaded


def suggest_registered_params(trial: optuna.Trial) -> dict[str, Any]:
    """Suggest optimizer parameters for all registered components."""
    out: dict[str, Any] = {}
    for component in registered_components():
        out[component.weight_name] = trial.suggest_float(
            component.weight_name,
            component.weight_min,
            component.weight_max,
        )
        for spec in component.params:
            out[spec.name] = spec.suggest(trial)
    return out


def registered_param_bounds() -> dict[str, tuple[float, float, bool]]:
    """Return numeric bounds for seed jitter and neighborhood search."""
    out: dict[str, tuple[float, float, bool]] = {}
    for component in registered_components():
        out[component.weight_name] = (
            component.weight_min,
            component.weight_max,
            False,
        )
        for spec in component.params:
            bound = spec.numeric_bound()
            if bound is not None:
                out[spec.name] = bound
    return out


def apply_registered_components(scorer: Any, params: dict[str, Any]) -> int:
    """Register runtime components on a ``ConfluenceScorer`` instance."""
    count = 0
    for component in registered_components():
        weight = float(params.get(component.weight_name, component.default_weight))
        if weight <= 0.0:
            continue
        scorer.register(component.name, component.make_scorer(params), weight)
        count += 1
    return count
