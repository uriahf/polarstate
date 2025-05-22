import polars as pl

def aalen_johansen(times: pl.Series, reals: pl.Series, event_of_interest: int = 1, competing_events: list[int] = [2]) -> pl.DataFrame:
    df = pl.DataFrame({"times": times, "reals": reals})
    df = df.sort("times")

    event_times = df.filter(pl.col("reals") != 0)["times"].unique().sort()
    result = []
    cum_hazard = 0.0

    for t in event_times:
        n_at_risk = df.filter(pl.col("times") >= t).height
        n_event = df.filter((pl.col("times") == t) & (pl.col("reals") == event_of_interest)).height

        if n_at_risk > 0:
            hazard = n_event / n_at_risk
            cum_hazard += hazard
            result.append({"time": t, "cuminc": cum_hazard})

    return pl.DataFrame(result)
