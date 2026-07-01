def _normalized_key(value):
    if isinstance(value, str):
        return value.strip().lower()
    return value


def is_unset_value(value):
    return isinstance(value, str) and value.strip().lower() == "unset"


def build_categorical_values(info):
    categories = info.get("enum_values")
    if not categories:
        categories = [info.get("min"), info.get("default"), info.get("max")]

    deduped = []
    seen = set()
    for value in categories:
        if value is None:
            continue
        key = _normalized_key(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def clamp_numeric_value(value, lower, upper, caster):
    try:
        numeric = caster(value)
    except Exception:
        return caster(upper)
    if numeric < lower:
        return caster(lower)
    if numeric > upper:
        return caster(upper)
    return numeric


def canonicalize_value(info, value, for_optimizer=False):
    ktype = str(info.get("type", "")).lower()

    if ktype == "integer":
        lower = int(info["min"])
        upper = int(info["max"])
        if is_unset_value(value):
            return upper if for_optimizer else "unset"
        return clamp_numeric_value(value, lower, upper, int)

    if ktype == "float":
        lower = float(info["min"])
        upper = float(info["max"])
        if is_unset_value(value):
            return upper if for_optimizer else "unset"
        return clamp_numeric_value(value, lower, upper, float)

    if ktype in ("enum", "boolean"):
        categories = build_categorical_values(info)
        if not categories:
            return value
        if value in categories:
            return value
        value_key = _normalized_key(value)
        for category in categories:
            if _normalized_key(category) == value_key:
                return category
        return categories[-1] if for_optimizer else value

    return value


def sanitize_config_for_optimizer(knobs_info, config):
    sanitized = {}
    for name, value in config.items():
        if name == "__unset__":
            continue
        info = knobs_info.get(name)
        if info is None:
            sanitized[name] = value
            continue
        sanitized[name] = canonicalize_value(info, value, for_optimizer=True)
    return sanitized
