"""RaySpec: the single source of truth for the obstacle ray fan.

Default comes from env.toml [obstacle_sensor]. Entry points may override it ONCE
(with_overrides) and hand the same object to env, detector, encoder, policy, tokenizer.
No other module may hard-code a ray count or ray range.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import tomllib

import navcore.configs


class RayConfigMismatchError(ValueError):
    """Two components disagree about the obstacle-ray configuration."""


@dataclass(slots=True, frozen=True)
class RaySpec:
    num_rays: int
    max_range: float
    fov_radians: float = 2.0 * math.pi

    def __post_init__(self) -> None:
        if self.num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {self.num_rays!r}.")
        if self.max_range <= 0.0:
            raise ValueError(f"max_range must be positive, got {self.max_range!r}.")

    def with_overrides(
        self, *, num_rays: int | None = None, max_range: float | None = None
    ) -> RaySpec:
        return replace(
            self,
            num_rays=self.num_rays if num_rays is None else num_rays,
            max_range=self.max_range if max_range is None else max_range,
        )


@lru_cache(maxsize=1)
def default_ray_spec() -> RaySpec:
    path = Path(navcore.configs.__file__).parent / "env.toml"
    with open(path, "rb") as f:
        table = tomllib.load(f)["obstacle_sensor"]
    return RaySpec(num_rays=int(table["num_rays"]), max_range=float(table["max_range"]))


def check_ray_counts(counts: Mapping[str, int], *, context: str) -> None:
    """First entry is the reference. Raises listing every component and its value."""
    ref_name, ref = next(iter(counts.items()))
    if all(v == ref for v in counts.values()):
        return
    lines = [
        f"Ray-count mismatch in {context}: components disagree on the number of obstacle rays.",
        f"  reference: '{ref_name}' = {ref}",
    ]
    lines += [f"  {'ok ' if v == ref else 'BAD'} {n} = {v}" for n, v in counts.items()]
    lines.append(
        "Every component must derive from one RaySpec "
        "(env.toml [obstacle_sensor], optionally overridden once at the entry point)."
    )
    raise RayConfigMismatchError("\n".join(lines))


def check_ray_specs(specs: Mapping[str, RaySpec], *, context: str) -> None:
    """Compares num_rays and max_range (fov is not compared)."""
    check_ray_counts({n: s.num_rays for n, s in specs.items()}, context=context)
    ref_name, ref = next(iter(specs.items()))
    bad = {
        n: s.max_range
        for n, s in specs.items()
        if not math.isclose(s.max_range, ref.max_range)
    }
    if bad:
        detail = "\n".join(f"  BAD {n}.max_range = {v}" for n, v in bad.items())
        raise RayConfigMismatchError(
            f"Ray-range mismatch in {context}: reference '{ref_name}' has "
            f"max_range={ref.max_range}, but:\n{detail}"
        )
