import copy
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Optional

import gymnasium as gym

from algorithms.base import Result
from algorithms.registry import CONTINUOUS_ALGOS, TABULAR_ALGOS, get_algorithm
from core.config import ALGO_DEFAULTS, ENV_DEFAULTS, AlgorithmConfig, EnvConfig
from environments.base import make_env


INT_PARAMS = {
    "early_stop_patience",
    "early_stop_min_delta",
    "solve_window",
    "max_episodes",
    "max_steps_per_episode",
    "batch_size",
    "ppo_epochs",
    "ppo_mini_batch_size",
    "rollout_steps",
    "target_update_freq",
    "buffer_capacity",
    "min_buffer_size",
    "n_discretization_bins",
    "seed",
    "log_interval",
}
FLOAT_PARAMS = {
    "lr",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
    "tau",
    "ppo_clip",
    "ppo_entropy_coef",
    "ppo_value_coef",
    "gae_lambda",
    "sac_alpha",
    "kl_target",
    "kl_beta",
    "sac_auto_alpha",
}


def _is_obs_discrete(env_name: str) -> bool:
    tmp = gym.make(env_name)
    result = isinstance(tmp.observation_space, gym.spaces.Discrete)
    tmp.close()
    return result


def _is_action_continuous(env_name: str) -> bool:
    tmp = gym.make(env_name)
    result = isinstance(tmp.action_space, gym.spaces.Box)
    tmp.close()
    return result


def run_experiment(
    algo_name: str,
    env_name: str,
    algo_config: Optional[AlgorithmConfig] = None,
    env_config: Optional[EnvConfig] = None,
) -> Result:
    if algo_config is None:
        base_name = algo_name.split(" (")[0]
        algo_config = copy.deepcopy(
            ALGO_DEFAULTS.get(base_name, AlgorithmConfig(algo_name=algo_name))
        )
    else:
        algo_config = copy.deepcopy(algo_config)
        algo_config.algo_name = algo_name

    base_name = algo_config.algo_name.split(" (")[0]

    if env_config is None:
        env_config = ENV_DEFAULTS.get(env_name, EnvConfig(env_name=env_name))

    train_fn = get_algorithm(base_name)
    obs_is_discrete = _is_obs_discrete(env_name)
    is_tabular = base_name in TABULAR_ALGOS

    env = make_env(
        env_name,
        discretize=(is_tabular and not obs_is_discrete),
        one_hot=(not is_tabular and obs_is_discrete),
    )

    if base_name in CONTINUOUS_ALGOS and not env.is_continuous:
        env.close()
        raise ValueError(
            f"{algo_name} requires a continuous action space, but {env_name} is discrete."
        )

    try:
        print(f"  [{algo_name}] running on {env_name}...")
        result = train_fn(env, algo_config)
        print(
            f"  [{algo_name}] done. Converged: {result.converged}, "
            f"Episodes to solve: {result.episodes_to_solve}, "
            f"Time: {result.wall_time:.1f}s"
        )
        return result
    except Exception as e:
        print(f"  [{algo_name}] FAILED: {e}")
        traceback.print_exc()
        return Result(
            algo_name=algo_name,
            env_name=env_name,
            config=asdict(algo_config),
            converged=False,
        )
    finally:
        env.close()


def _set_param(cfg, param_name, val):
    if param_name in INT_PARAMS:
        setattr(cfg, param_name, int(float(val)))
    elif param_name in FLOAT_PARAMS:
        setattr(cfg, param_name, float(val))
    elif param_name == "hidden_dims":
        setattr(cfg, param_name, tuple(eval(val)))
    else:
        setattr(cfg, param_name, type(getattr(cfg, param_name))(val))


def parse_variants(variant_str: str) -> list[tuple[str, AlgorithmConfig]]:
    variants = []
    for spec in variant_str.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            algo_name = spec
            variants.append(
                (
                    algo_name,
                    copy.deepcopy(
                        ALGO_DEFAULTS.get(algo_name, AlgorithmConfig(algo_name=algo_name))
                    ),
                )
            )
            continue
        algo_name, params_str = spec.split(":", 1)
        base = copy.deepcopy(ALGO_DEFAULTS.get(algo_name, AlgorithmConfig(algo_name=algo_name)))
        param_name, values_str = params_str.split("=", 1)
        values = [v.strip() for v in values_str.split(",")]
        for val in values:
            cfg = copy.deepcopy(base)
            _set_param(cfg, param_name, val)
            label = f"{algo_name} ({param_name}={val})"
            cfg.algo_name = label
            variants.append((label, cfg))
    return variants


def parse_sweep(sweep_str: str) -> list[tuple[str, AlgorithmConfig]]:
    groups = []
    for spec in sweep_str.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        algo_name, params_str = spec.split(":", 1)
        base = copy.deepcopy(ALGO_DEFAULTS.get(algo_name, AlgorithmConfig(algo_name=algo_name)))

        param_specs = []
        depth = 0
        current = []
        for ch in params_str:
            if ch == "[":
                depth += 1
                current.append(ch)
            elif ch == "]":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                param_specs.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            param_specs.append("".join(current).strip())

        parsed_params = []
        for ps in param_specs:
            pn, vs = ps.split("=", 1)
            vs = vs.strip()
            if vs.startswith("[") and vs.endswith("]"):
                vals = [v.strip() for v in vs[1:-1].split(",")]
            else:
                vals = [vs]
            parsed_params.append((pn, vals))

        def generate(i, current_cfg, variants):
            if i == len(parsed_params):
                parts = []
                for pn, vals in parsed_params:
                    parts.append(f"{pn}={getattr(current_cfg, pn)}")
                label = f"{algo_name} ({', '.join(parts)})"
                cfg = copy.deepcopy(current_cfg)
                cfg.algo_name = label
                variants.append((label, cfg))
                return
            pn, vals = parsed_params[i]
            for v in vals:
                new_cfg = copy.deepcopy(current_cfg)
                _set_param(new_cfg, pn, v)
                generate(i + 1, new_cfg, variants)

        variants = []
        generate(0, base, variants)
        groups.extend(variants)
    return groups


def _build_experiment_list(
    env_names, algo_names, variants_str, sweep_str, max_episodes
) -> list[tuple[str, str, AlgorithmConfig, EnvConfig]]:
    experiments = []

    for env_name in env_names:
        is_cont = _is_action_continuous(env_name)
        env_cfg = ENV_DEFAULTS.get(env_name, EnvConfig(env_name=env_name))

        if sweep_str:
            specs = parse_sweep(sweep_str)
        elif variants_str:
            specs = parse_variants(variants_str)
        else:
            specs = []
            for algo_name in algo_names or []:
                if algo_name in CONTINUOUS_ALGOS and not is_cont:
                    continue
                cfg = copy.deepcopy(
                    ALGO_DEFAULTS.get(algo_name, AlgorithmConfig(algo_name=algo_name))
                )
                specs.append((algo_name, cfg))

        for label, cfg in specs:
            base = label.split(" (")[0]
            if base in CONTINUOUS_ALGOS and not is_cont:
                print(f"  Skipping {label} on {env_name} (not continuous)")
                continue
            if max_episodes is not None:
                cfg.max_episodes = max_episodes
            experiments.append((label, env_name, cfg, env_cfg))

    return experiments


def run_all_algorithms(
    env_names: list[str],
    algo_names: Optional[list[str]] = None,
    algo_config: Optional[AlgorithmConfig] = None,
    max_episodes: Optional[int] = None,
    variants_str: Optional[str] = None,
    sweep_str: Optional[str] = None,
    workers: int = 0,
) -> dict[str, dict[str, Result]]:
    if algo_names is None and variants_str is None and sweep_str is None:
        from algorithms.registry import list_algorithms

        algo_names = list_algorithms()

    experiments = _build_experiment_list(
        env_names, algo_names, variants_str, sweep_str, max_episodes
    )

    if not experiments:
        return {}

    results: dict[str, dict[str, Result]] = {}

    if workers > 1:
        print(f"  Running {len(experiments)} experiments with {workers} workers...")
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(run_experiment, label, env_name, cfg, env_cfg): (label, env_name)
                for label, env_name, cfg, env_cfg in experiments
            }
            for future in as_completed(futures):
                label, env_name = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"  [{label}] FAILED in worker: {e}")
                    result = Result(algo_name=label, env_name=env_name, converged=False)
                results.setdefault(env_name, {})[label] = result
    else:
        for label, env_name, cfg, env_cfg in experiments:
            if env_name not in results:
                results[env_name] = {}
            result = run_experiment(label, env_name, algo_config=cfg, env_config=env_cfg)
            results[env_name][label] = result

    return results
