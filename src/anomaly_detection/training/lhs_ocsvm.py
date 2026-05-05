"""Latin Hypercube samples for RBF OCSVM grids (experimental plan §2.1 LHS note)."""



from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True, slots=True)
class LHSOCSVMConfig:
    """One decoded `(nu, gamma)` pairing for salted RBF grids."""

    nu: float
    gamma: float | str


def lhs_rbf_ocsvm_configs(
    n_samples: int = 80,
    seed: int = 42,
) -> tuple[LHSOCSVMConfig, ...]:
    """Draw ``n_samples`` RBF LHS draws mapping plan §2.1 ``nu``-by-``gamma`` core axes.

    Categorical kernels are handled elsewhere (oracle combinatorial). This routine
    implements the complementary **primary** ``nu`` / ``gamma`` exploration at scale.
    """
    lh = qmc.LatinHypercube(d=2, seed=seed)
    unit = lh.random(n=n_samples)

    nus = np.array([0.01, 0.05, 0.10, 0.20, 0.35, 0.50], dtype=np.float64)
    categorical_gammas: tuple[float | str, ...] = (
        "scale",
        "auto",
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    )
    bins_nu = nus.shape[0]
    bins_gamma = len(categorical_gammas)

    configs: list[LHSOCSVMConfig] = []
    for row in unit:
        nu_index = min(int(row[0] * bins_nu), bins_nu - 1)
        gamma_index = min(int(row[1] * bins_gamma), bins_gamma - 1)
        configs.append(
            LHSOCSVMConfig(
                nu=float(nus[nu_index]),
                gamma=categorical_gammas[gamma_index],
            )
        )
    return tuple(configs)


def lhs_dict_rows(n_samples: int = 80, seed: int = 42) -> tuple[dict[str, Any], ...]:
    """Return ``OCSVMModel`` kwargs rows with ``kernel="rbf"`` pinned."""
    return tuple(
        {"nu": cfg.nu, "gamma": cfg.gamma}
        for cfg in lhs_rbf_ocsvm_configs(
            n_samples=n_samples,
            seed=seed,
        )
    )
