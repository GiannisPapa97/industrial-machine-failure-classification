"""The error raised for a config that does not make sense."""

from machine_failure.cli import UserError


class ConfigError(UserError, ValueError):
    """A config value is missing, of the wrong type or out of range."""
