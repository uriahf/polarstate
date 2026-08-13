---
name: polarstate
description: >
  Fast Aalen-Johansen state-occupancy estimates for Polars DataFrames. Use when writing Python code that uses the polarstate package.
compatibility: Requires Python >=3.9.
---

# Polars State

Fast Aalen-Johansen state-occupancy estimates for Polars DataFrames

## Installation

```bash
pip install polarstate
```

## API overview

### Event table

Compute an inspectable Aalen-Johansen event table from time-to-event observations.

- `prepare_event_table`: Compute an inspectable Aalen-Johansen event table

### Time-horizon estimates

Return state-occupancy estimates at requested time horizons.

- `predict_aj_estimates`: Evaluate state-occupancy probabilities at fixed time horizons

## Resources

- [Full documentation](https://uriahf.github.io/polarstate/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
- [Source code](https://github.com/uriahf/polarstate)
