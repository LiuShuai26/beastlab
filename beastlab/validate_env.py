"""Validator for Beast GameEnv .so modules.

Imports a pybind11 GameEnv module and runs a series of checks to verify
it exposes the expected API and behaves correctly.

Usage:
    python -m beastlab.validate_env HumanoidEnv
    python -m beastlab.validate_env HumanoidEnv --project path/to/Project.project
    python -m beastlab.validate_env HumanoidEnv --steps 200
"""

import argparse
import sys
import traceback

import numpy as np

from beastlab.env_loader import load_beast_env


def _check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return passed


def validate(module_name, project_path=None, num_steps=100):
    """Run validation checks on a Beast GameEnv .so module.

    Returns True if all checks pass, False otherwise.
    """
    results = []
    print(f"Validating module: {module_name}")
    print("=" * 60)

    # --- 1. Import module ---
    print("\n[Import]")
    try:
        module = load_beast_env(module_name)
        results.append(_check("import module", True))
    except Exception as e:
        _check("import module", False, str(e))
        return False

    # --- 2. Instantiate env ---
    print("\n[Instantiation]")
    try:
        EnvClass = getattr(module, module_name)
        results.append(_check("find env class", True, f"{module_name}.{module_name}"))
    except AttributeError:
        _check("find env class", False, f"no class '{module_name}' in module")
        return False

    try:
        if project_path:
            env = EnvClass(project_path)
        else:
            env = EnvClass()
        results.append(_check("instantiate env", True))
    except Exception as e:
        _check("instantiate env", False, str(e))
        return False

    # --- 3. Check API methods exist ---
    print("\n[API Methods]")
    required_methods = ["reset", "step", "observation_size", "action_space"]
    optional_methods = ["set_fixed_timestep", "set_physics_substeps", "load_from_string"]
    for m in required_methods:
        has = hasattr(env, m) and callable(getattr(env, m))
        results.append(_check(f"has {m}()", has))
    for m in optional_methods:
        has = hasattr(env, m) and callable(getattr(env, m))
        _check(f"has {m}() [optional]", has)

    # --- 4. observation_size / action_space ---
    print("\n[Spaces]")
    try:
        obs, info = env.reset()
        results.append(_check("reset() returns (obs, info)", True))
    except Exception as e:
        _check("reset()", False, str(e))
        return False

    obs = np.asarray(obs, dtype=np.float32)
    obs_size = env.observation_size()
    results.append(_check(
        "observation_size matches obs",
        obs.shape == (obs_size,),
        f"obs.shape={obs.shape}, observation_size={obs_size}",
    ))
    results.append(_check("obs dtype is float32", obs.dtype == np.float32, str(obs.dtype)))
    results.append(_check("obs has no NaN", not np.any(np.isnan(obs))))
    results.append(_check("obs has no Inf", not np.any(np.isinf(obs))))

    action_info = env.action_space()
    results.append(_check("action_space() returns dict", isinstance(action_info, dict)))
    cont_size = action_info.get("continuous_size", 0)
    discrete_branches = action_info.get("discrete_branch_sizes", [])
    has_actions = cont_size > 0 or len(discrete_branches) > 0
    results.append(_check(
        "env has actions defined",
        has_actions,
        f"continuous={cont_size}, discrete_branches={discrete_branches}",
    ))

    # --- 5. Step with random actions ---
    print(f"\n[Step Loop] running {num_steps} steps with random actions")
    action_dim = cont_size + len(discrete_branches)
    if action_dim == 0:
        _check("step loop", False, "no actions to send")
        return False

    step_ok = True
    nan_obs_step = None
    inf_obs_step = None
    reset_count = 0
    try:
        for i in range(num_steps):
            # Build action: continuous part in [-1,1], discrete part as random ints
            parts = []
            if cont_size > 0:
                parts.append(np.random.uniform(-1, 1, size=(cont_size,)).astype(np.float32))
            for branch_size in discrete_branches:
                parts.append(np.array([np.random.randint(0, branch_size)], dtype=np.float32))
            action = np.concatenate(parts) if len(parts) > 1 else parts[0]

            result = env.step(action)
            if len(result) != 5:
                _check("step returns 5-tuple", False, f"got {len(result)} values at step {i}")
                step_ok = False
                break

            obs, rewards, terminated, truncated, info = result
            obs = np.asarray(obs, dtype=np.float32)

            if np.any(np.isnan(obs)) and nan_obs_step is None:
                nan_obs_step = i
            if np.any(np.isinf(obs)) and inf_obs_step is None:
                inf_obs_step = i

            done = bool(terminated[0]) if hasattr(terminated, '__len__') else bool(terminated)
            trunc = bool(truncated[0]) if hasattr(truncated, '__len__') else bool(truncated)
            if done or trunc:
                reset_count += 1
                obs, info = env.reset()
    except Exception as e:
        _check("step loop", False, f"exception at step {i}: {e}")
        traceback.print_exc()
        step_ok = False

    if step_ok:
        results.append(_check("step loop completed", True, f"{num_steps} steps, {reset_count} resets"))
    results.append(_check("no NaN in obs during steps", nan_obs_step is None,
                          "" if nan_obs_step is None else f"first NaN at step {nan_obs_step}"))
    results.append(_check("no Inf in obs during steps", inf_obs_step is None,
                          "" if inf_obs_step is None else f"first Inf at step {inf_obs_step}"))

    # --- 6. Reset with seed ---
    print("\n[Reset with seed]")
    try:
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        obs1 = np.asarray(obs1, dtype=np.float32)
        obs2 = np.asarray(obs2, dtype=np.float32)
        deterministic = np.allclose(obs1, obs2)
        _check("reset(seed=42) is deterministic [info]", deterministic,
               "same seed produces same obs" if deterministic else "non-deterministic (may be ok)")
    except Exception as e:
        _check("reset(seed=) supported", False, str(e))

    # --- Summary ---
    passed = sum(results)
    total = len(results)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f", {failed} FAILED")
    else:
        print(" -- all checks passed!")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Validate a Beast GameEnv .so module")
    parser.add_argument("module", help="Module name (e.g. HumanoidEnv)")
    parser.add_argument("--project", default=None, help="Path to .project file (if required by env)")
    parser.add_argument("--steps", type=int, default=100, help="Number of random steps to run (default: 100)")
    args = parser.parse_args()

    success = validate(args.module, project_path=args.project, num_steps=args.steps)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
