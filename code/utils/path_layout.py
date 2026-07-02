import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root():
    current = Path(__file__).resolve().parent
    candidates = [current] + list(current.parents)
    for candidate in candidates:
        if (candidate / "main.py").exists() and (candidate / "config").exists():
            return str(candidate)
    return str(Path(__file__).resolve().parents[1])


def resolve_repo_path(path_value, base_dir=None, prefer_existing=True):
    if path_value is None:
        return path_value

    raw = str(path_value).strip()
    if not raw:
        return raw

    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if expanded.is_absolute():
        return str(expanded)

    candidates = []
    if base_dir:
        candidates.append(Path(base_dir) / expanded)
    candidates.append(Path(repo_root()) / expanded)

    if prefer_existing:
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve())

    return str(candidates[-1].resolve())


def resolve_repo_script(path_value, base_dir=None):
    if path_value is None:
        return path_value

    raw = str(path_value).strip()
    if not raw:
        return raw

    if os.path.isabs(raw):
        return raw

    if "/" not in raw and "\\" not in raw and "." not in raw:
        return raw

    resolved = resolve_repo_path(raw, base_dir=base_dir, prefer_existing=True)
    return resolved if os.path.exists(resolved) else raw


def default_discovery_rule_file(sys_name):
    return resolve_repo_path(f"config/dependency_discovery/{sys_name}_rules.json", prefer_existing=False)


def default_online_rule_file(sys_name):
    return resolve_repo_path(f"config/dependency_aware/{sys_name}_online_rules.json", prefer_existing=False)
