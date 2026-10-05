"""Replayable forecasting environment."""

__all__ = ["EpisodeResult", "ForecastConfig", "ForecastEnv", "ForecastMode", "ForecastStep"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import env

    value = getattr(env, name)
    globals()[name] = value
    return value
