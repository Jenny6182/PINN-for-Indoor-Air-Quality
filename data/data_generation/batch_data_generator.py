from itertools import product
from concurrent.futures import ProcessPoolExecutor
import constant, varying

def _run(cfg):
    if isinstance(cfg, constant.ConstantConfig):
        return constant.run(cfg)
    elif isinstance(cfg, varying.VaryingConfig):
        return varying.run(cfg)
    else:
        raise ValueError(f"Unknown config type: {type(cfg)}")

if __name__ == "__main__":
    configs = []

    # --- constant runs ---
    configs += [
        constant.ConstantConfig(Q=q, S_vol=s)
        for q, s in product([100, 200, 500], [0.05, 0.10, 0.30])
    ]

    # --- varying Q runs ---
    configs += [
        varying.VaryingConfig(mode="Q", seed=s, n_segments=n)
        for s, n in product(range(3), [8, 16])
    ]

    # --- varying S runs ---
    configs += [
        varying.VaryingConfig(mode="S", seed=s, n_segments=n)
        for s, n in product(range(3), [8, 16])
    ]

    with ProcessPoolExecutor() as ex:
        list(ex.map(_run, configs))