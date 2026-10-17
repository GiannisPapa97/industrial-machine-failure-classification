"""Read a config from a TOML file.

The file holds the fields of one config as top-level keys, with the nested data
settings in a ``[data]`` table. Keys that are missing keep their default; keys that
do not exist, or have the wrong type, are an error that names the key.
"""

import dataclasses
import tomllib
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

from machine_failure.config.errors import ConfigError


def convert(value: Any, hint: Any, key: str) -> Any:
    """Turn a TOML ``value`` into the type ``hint``, or raise ``ConfigError``."""
    origin = get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        return build(hint, value, key)
    if origin is tuple:
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a list, not {value!r}")
        (item_hint, _) = get_args(hint)
        return tuple(convert(v, item_hint, f"{key}[{i}]") for i, v in enumerate(value))
    if origin is Literal or hint is str:
        expected: type = str
    elif hint is Path:
        return Path(convert(value, str, key))
    elif hint is float:
        # an integer is fine where a float is expected (0 or 1), a bool is not
        if isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        expected = float
    else:
        expected = hint
    if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
        raise ConfigError(f"{key} must be {expected.__name__}, not {value!r}")
    return value


def build[C](config_class: type[C], table: Any, key: str = "config") -> C:
    """Construct ``config_class`` from a TOML table, checking keys and types."""
    if not isinstance(table, dict):
        raise ConfigError(f"{key} must be a table, not {table!r}")
    hints = get_type_hints(config_class)
    fields = {f.name for f in dataclasses.fields(config_class)}  # type: ignore[arg-type]
    unknown = sorted(set(table) - fields)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {key}: {', '.join(unknown)}; known: {sorted(fields)}"
        )
    values = {
        name: convert(value, hints[name], name if key == "config" else f"{key}.{name}")
        for name, value in table.items()
    }
    return config_class(**values)


def load_config[C](path: Path, config_class: type[C]) -> C:
    """Read the TOML file at ``path`` as a ``config_class``."""
    try:
        table = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from error
    try:
        return build(config_class, table)
    except ConfigError as error:
        raise ConfigError(f"{path}: {error}") from error
