import json

from utils.path_layout import resolve_repo_path


class BaseSystemAdapter:
    """Shared helper methods for all target-system adapters."""

    @staticmethod
    def load_knobs(knob_config_file, knob_num, normalize=None):
        resolved_path = resolve_repo_path(knob_config_file, prefer_existing=True)
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if normalize is not None:
            raw = normalize(raw)
        if knob_num == -1:
            return raw
        keys = list(raw.keys())[:knob_num]
        return {k: raw[k] for k in keys}

    @staticmethod
    def build_default_knobs(knobs_info):
        default_knobs = {}
        for name, value in knobs_info.items():
            if value.get("type") != "combination":
                default_knobs[name] = value.get("default")
                continue
            knob_names = name.strip().split("|")
            knob_values = str(value.get("default", "")).strip().split("|")
            for idx, knob_name in enumerate(knob_names):
                if idx < len(knob_values):
                    try:
                        default_knobs[knob_name] = int(knob_values[idx])
                    except Exception:
                        default_knobs[knob_name] = knob_values[idx]
        return default_knobs

    def get_default_knobs(self):
        return dict(getattr(self, "default_knobs", {}))

    def collect_runtime_evidence(self, since=None, tail=200):
        """Optional hook for system-level evidence beyond workload stdout/stderr."""
        return []
