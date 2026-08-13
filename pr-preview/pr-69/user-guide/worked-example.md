# Worked Example

This tutorial follows a small cohort from subject-level observations to state-occupancy estimates. The code is executed when the documentation builds, so it also acts as a continuously checked example.


# Create the observations

Each row contains an observed time and an outcome code. Code `0` means follow-up ended without an observed event (right-censoring), code `1` is the event of interest, and optional code `2` is a competing event.


``` python
import polars as pl
from great_docs import tbl_explorer
from polarstate import prepare_event_table, predict_aj_estimates

observations = pl.DataFrame(
    {
        "times": [4.3, 9.7, 14.2, 18.6, 24.1, 31.5, 34.8, 39.2, 46.0, 49.9],
        "reals": [1, 1, 2, 1, 1, 0, 0, 1, 2, 1],
    }
)
observations
```


shape: (10, 2)

| times | reals |
|-------|-------|
| f64   | i64   |
| 4.3   | 1     |
| 9.7   | 1     |
| 14.2  | 2     |
| 18.6  | 1     |
| 24.1  | 1     |
| 31.5  | 0     |
| 34.8  | 0     |
| 39.2  | 1     |
| 46.0  | 2     |
| 49.9  | 1     |


# Prepare the event table


``` python
event_table = prepare_event_table(observations)
event_table_view = event_table.select(
    "times",
    "at_risk",
    "count_0",
    "count_1",
    "count_2",
    "overall_survival",
    "state_occupancy_probability_1_at_times",
    "state_occupancy_probability_2_at_times",
)
```


Explore the calculation row by row. Sort by time, hide intermediate columns, or copy the visible values for comparison.


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;600&display=swap');
#gd-tbl-5ea5bc9b .gt_table {
  display: table;
  border-collapse: collapse;
  table-layout: fixed;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 14px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
}
#gd-tbl-5ea5bc9b .gt_heading {
  background-color: #FFFFFF;
  text-align: left;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-right-style: none;
}
#gd-tbl-5ea5bc9b .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}
#gd-tbl-5ea5bc9b .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}
#gd-tbl-5ea5bc9b .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
#gd-tbl-5ea5bc9b .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-right-style: none;
}
#gd-tbl-5ea5bc9b .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: solid;
  border-left-width: 1px;
  border-left-color: #F2F2F2;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #F2F2F2;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}
#gd-tbl-5ea5bc9b .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}
#gd-tbl-5ea5bc9b .gt_row {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #E9E9E9;
  border-left-style: solid;
  border-left-width: 1px;
  border-left-color: #E9E9E9;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #E9E9E9;
  vertical-align: middle;
  overflow-x: hidden;
}
#gd-tbl-5ea5bc9b .gt_left { text-align: left; }
#gd-tbl-5ea5bc9b .gt_center { text-align: center; }
#gd-tbl-5ea5bc9b .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
#gd-tbl-5ea5bc9b .gt_font_normal { font-weight: normal; }
#gd-tbl-5ea5bc9b .gt_font_bold { font-weight: bold; }
#gd-tbl-5ea5bc9b .gt_font_italic { font-style: italic; }
#gd-tbl-5ea5bc9b .gt_striped { background-color: rgba(128,128,128,0.05); }
#gd-tbl-5ea5bc9b .gt_from_md > :first-child { margin-top: 0; }
#gd-tbl-5ea5bc9b .gt_from_md > :last-child { margin-bottom: 0; }
/* Data cell font */
#gd-tbl-5ea5bc9b .gt_table_body .gt_row {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #333333;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 14px;
}
/* Column label font */
#gd-tbl-5ea5bc9b .gt_col_heading {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #333333;
}
/* Row number styling */
#gd-tbl-5ea5bc9b .gd-tbl-rownum {
  color: gray;
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  border-right: 2px solid rgba(102, 153, 204, 0.5) !important;
  text-align: right;
  padding-right: 6px;
  white-space: nowrap;
}
/* Head/tail divider */
#gd-tbl-5ea5bc9b tr.gd-tbl-divider td,
#gd-tbl-5ea5bc9b tr.gd-tbl-divider th {
  border-bottom: 2px solid rgba(102, 153, 204, 0.5);
}
/* Missing values */
#gd-tbl-5ea5bc9b .gd-tbl-missing {
  color: #B22222 !important;
  background-color: rgba(255, 193, 193, 0.35);
}
/* Column label sub-elements */
#gd-tbl-5ea5bc9b .gd-tbl-colname {
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  padding-bottom: 0;
  margin-bottom: 0;
}
#gd-tbl-5ea5bc9b .gd-tbl-dtype {
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  padding-top: 0;
  margin-top: 0;
  color: #666666;
}
/* Header badge styling */
#gd-tbl-5ea5bc9b .gd-tbl-badge {
  display: inline-block;
  padding: 2px 10px;
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 10px;
  font-weight: bold;
  text-transform: uppercase;
  position: inherit;
}
#gd-tbl-5ea5bc9b .gd-tbl-badge-rows-label {
  background-color: #eecbff;
  color: #333333;
  border: 1px solid #eecbff;
  margin-left: 5px;
}
#gd-tbl-5ea5bc9b .gd-tbl-badge-rows-value {
  background-color: transparent;
  color: #333333;
  border: 1px solid #eecbff;
  margin-left: -4px;
  margin-right: 3px;
}
#gd-tbl-5ea5bc9b .gd-tbl-badge-cols-label {
  background-color: #BDE7B4;
  color: #333333;
  border: 1px solid #BDE7B4;
}
#gd-tbl-5ea5bc9b .gd-tbl-badge-cols-value {
  background-color: transparent;
  color: #333333;
  border: 1px solid #BDE7B4;
  margin-left: -4px;
}
/* Dark mode */
body.quarto-dark #gd-tbl-5ea5bc9b .gt_table,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_table,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_table {
  background-color: #1a1a2e;
  color: #e0e0e0;
  border-top-color: #555;
  border-bottom-color: #555;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_heading,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_heading,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_heading {
  background-color: #1a1a2e;
  border-bottom-color: #1a1a2e;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_title,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_title,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_title {
  color: #e0e0e0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_subtitle,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_subtitle,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_subtitle {
  color: #b0b0b0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_col_headings,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_col_headings,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_col_headings {
  border-top-color: #444;
  border-bottom-color: #444;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_col_heading,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_col_heading,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_col_heading {
  color: #b0b0b0;
  background-color: #1a1a2e;
  border-left-color: #333;
  border-right-color: #333;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_table_body,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_table_body,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_table_body {
  border-top-color: #444;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_table_body .gt_row,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_table_body .gt_row,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_table_body .gt_row {
  color: #d0d0d0;
  border-top-color: #4b4b4b;
  border-bottom-color: #4b4b4b;
  border-left-color: #333;
  border-right-color: #333;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-rownum,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-rownum,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-rownum {
  color: #888;
  border-right-color: rgba(102, 153, 204, 0.4) !important;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-missing,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-missing,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-missing {
  color: #ff6b6b !important;
  background-color: rgba(60, 17, 24, 0.2);
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-dtype,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-dtype,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-dtype {
  color: #888;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gt_bottom_border,
html.quarto-dark #gd-tbl-5ea5bc9b .gt_bottom_border,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gt_bottom_border {
  border-bottom-color: #555;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-label,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-label,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-label {
  background-color: #3d2a4d;
  border-color: #3d2a4d;
  color: #e0c0ff;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-value,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-value,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-badge-rows-value {
  border-color: #3d2a4d;
  color: #d0b0e0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-label,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-label,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-label {
  background-color: #2a4d2a;
  border-color: #2a4d2a;
  color: #b0e0b0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-value,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-value,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-badge-cols-value {
  border-color: #2a4d2a;
  color: #a0d0a0;
}
</style> <style>
/* ── Table scroll wrapper ────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  width: 100%;
}
/* ── Toolbar ─────────────────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px 0;
  align-items: center;
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  font-size: 13px;
}
/* ── Filter bar ──────────────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-filter-bar {
  flex: 1 1 200px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  min-height: 30px;
  padding: 3px 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  position: relative;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-tokens {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-token {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 2px 4px 2px 8px;
  background: #e8f0fe;
  border: 1px solid #c4d9f2;
  border-radius: 12px;
  font-size: 11px;
  color: #1a3a5c;
  white-space: nowrap;
  max-width: 260px;
  line-height: 1.4;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-token-text {
  overflow: hidden;
  text-overflow: ellipsis;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-token-x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  background: #d0dfef;
  color: #4a6a8a;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 50%;
  padding: 0;
  padding-bottom: 2px;
  line-height: 1;
  flex-shrink: 0;
  transition: background 0.1s, color 0.1s;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-token-x:hover {
  background: #a0bdd8;
  color: #1a3a5c;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-token-case {
  font-size: 9px;
  font-weight: 700;
  color: #4477aa;
  border: 1px solid #a0bdd8;
  border-radius: 3px;
  padding: 0 3px;
  line-height: 1.4;
  flex-shrink: 0;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-add {
  flex-shrink: 0;
  border: none;
  background: none;
  padding: 3px;
  color: #6699CC;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-add:hover {
  background: #eef3fb;
  border-radius: 3px;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-hint {
  color: #b0b0b0;
  font-size: 12px;
  font-style: italic;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding-left: 4px;
  user-select: none;
  pointer-events: none;
}
#gd-tbl-5ea5bc9b .gd-tbl-filter-hint svg {
  flex-shrink: 0;
  stroke: #b0b0b0;
}
/* ── Filter wizard dropdown ──────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-filter-wizard {
  position: absolute;
  top: calc(100% + 2px);
  left: 0;
  z-index: 200;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  padding: 8px 0;
  min-width: 200px;
  max-width: 320px;
  max-height: 300px;
  overflow-y: auto;
  font-size: 12px;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-label {
  display: block;
  padding: 4px 12px 4px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #888;
  font-weight: 600;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-options {
  display: flex;
  flex-direction: column;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5px 12px;
  border: none;
  background: none;
  text-align: left;
  font-size: 12px;
  font-family: inherit;
  color: #333;
  cursor: pointer;
  transition: background 0.1s;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-option:hover {
  background: #f0f4fb;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-dtype {
  font-size: 9px;
  color: #999;
  background: #f0f0f0;
  padding: 1px 5px;
  border-radius: 3px;
  margin-left: 8px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-input {
  margin: 4px 12px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
  font-family: inherit;
  background: #fff;
  color: #333;
  outline: none;
  width: calc(100% - 24px);
  box-sizing: border-box;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-input:focus {
  border-color: #6699CC;
  box-shadow: 0 0 0 2px rgba(102,153,204,0.2);
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-between {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 4px;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-between .gd-tbl-fw-input {
  flex: 1;
  margin: 4px;
  min-width: 60px;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-sep {
  font-size: 11px;
  color: #888;
  flex-shrink: 0;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-commit {
  margin: 4px 12px 6px;
  font-size: 11px;
  padding: 4px 14px;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-input-row {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 8px;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-input-row .gd-tbl-fw-input {
  flex: 1;
  margin: 4px 0;
  width: auto;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-case {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 26px;
  margin-left: 4px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f8f8f8;
  color: #999;
  font-size: 11px;
  font-weight: 700;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.15s;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-case:hover {
  border-color: #999;
  color: #666;
}
#gd-tbl-5ea5bc9b .gd-tbl-fw-case.active {
  background: #e0edff;
  border-color: #6699CC;
  color: #336699;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f8f8f8;
  color: #333;
  font-size: 12px;
  font-weight: 500;
  font-family: inherit;
  line-height: 14px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn:hover {
  background: #eee;
  border-color: #aaa;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn:focus-visible {
  outline: 2px solid #6699CC;
  outline-offset: 1px;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn-active {
  background: #e0edff;
  border-color: #6699CC;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn-icon {
  padding: 5px 7px;
  line-height: 0;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn-icon svg {
  display: block;
}
/* Copy-success green checkmark state */
#gd-tbl-5ea5bc9b .gd-tbl-btn-copied {
  color: #198754;
  border-color: #198754;
}
/* ── Button wrapper + tooltip ────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-btn-wrap {
  position: relative;
  display: inline-block;
}
#gd-tbl-5ea5bc9b .gd-tbl-tooltip {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  top: calc(100% + 4px);
  left: 50%;
  transform: translateX(-50%);
  padding: 3px 8px;
  background: #333;
  color: #fff;
  border-radius: 3px;
  font-size: 11px;
  white-space: nowrap;
  pointer-events: none;
  transition: opacity 0.15s;
  z-index: 100;
}
/* Keep tooltip from overflowing right edge */
#gd-tbl-5ea5bc9b .gd-tbl-btn-wrap:last-child .gd-tbl-tooltip {
  left: auto;
  right: 0;
  transform: none;
}
#gd-tbl-5ea5bc9b .gd-tbl-btn-wrap:hover .gd-tbl-tooltip {
  visibility: visible;
  opacity: 1;
}
/* ── Column toggle dropdown ──────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-col-wrap {
  position: relative;
  display: inline-block;
}
#gd-tbl-5ea5bc9b .gd-tbl-col-wrap .gd-tbl-tooltip {
  left: auto;
  right: 0;
  transform: none;
}
#gd-tbl-5ea5bc9b .gd-tbl-col-menu {
  display: none;
  position: absolute;
  top: 100%;
  right: 0;
  z-index: 10;
  min-width: 180px;
  max-height: 300px;
  overflow-y: auto;
  margin-top: 4px;
  padding: 6px 0;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
#gd-tbl-5ea5bc9b .gd-tbl-col-menu.open {
  display: block;
}
#gd-tbl-5ea5bc9b .gd-tbl-col-option {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  cursor: pointer;
  font-size: 12px;
  user-select: none;
}
#gd-tbl-5ea5bc9b .gd-tbl-col-option:hover {
  background: #f0f0f0;
}
/* ── Sort indicators ─────────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-sortable {
  cursor: pointer;
  user-select: none;
  position: relative;
}
#gd-tbl-5ea5bc9b .gd-tbl-sort-icon {
  display: inline-block;
  width: 10px;
  height: 14px;
  margin-left: 4px;
  color: #bbb;
  vertical-align: middle;
}
#gd-tbl-5ea5bc9b .gd-tbl-sort-icon svg {
  display: block;
  width: 10px;
  height: 14px;
  fill: currentColor;
}
#gd-tbl-5ea5bc9b .gd-tbl-sort-asc .gd-tbl-sort-icon,
#gd-tbl-5ea5bc9b .gd-tbl-sort-desc .gd-tbl-sort-icon {
  color: #6699CC;
}
/* ── Search highlight ────────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-highlight {
  background-color: #FFEEBA;
  border-radius: 2px;
  padding: 0 1px;
}
/* ── Pagination ──────────────────────────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-pagination {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 0;
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  font-size: 12px;
  color: #666;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-info {
  white-space: nowrap;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-nav {
  display: flex;
  gap: 2px;
  align-items: center;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  padding: 0 6px;
  border: 1px solid #ddd;
  border-radius: 3px;
  background: #fff;
  color: #333;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  transition: background 0.1s;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-btn:hover {
  background: #f0f0f0;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-btn.active {
  background: #6699CC;
  color: #fff;
  border-color: #6699CC;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
#gd-tbl-5ea5bc9b .gd-tbl-page-ellipsis {
  padding: 0 4px;
  color: #999;
}
/* ── Dark mode ───────────────────────────────────── */
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-bar,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-bar,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-bar {
  background-color: #2a2a3e;
  border-color: #444;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-token {
  background: #2d3a50;
  border-color: #3d5070;
  color: #b0ccee;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token-x:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token-x:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-token-x:hover {
  background: #3d5070;
  color: #e0e8f0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token-case,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-token-case,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-token-case {
  color: #88bbee;
  border-color: #4d6888;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-case {
  background: #2a2a3e;
  border-color: #555;
  color: #888;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-case:hover {
  border-color: #888;
  color: #bbb;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case.active,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-case.active,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-case.active {
  background: #2d3a50;
  border-color: #6699CC;
  color: #88bbee;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-add,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-add,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-add {
  color: #88bbee;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-add:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-add:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-add:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-hint,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-hint,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-hint {
  color: #666;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-hint svg,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-hint svg,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-hint svg {
  stroke: #666;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-wizard,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-filter-wizard,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-filter-wizard {
  background: #1e1e32;
  border-color: #444;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-option,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-option,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-option {
  color: #ddd;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-option:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-option:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-option:hover {
  background: #2a2a44;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-dtype,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-dtype,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-dtype {
  background: #333;
  color: #aaa;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-input,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-input,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-input {
  background: #2a2a3e;
  border-color: #555;
  color: #e0e0e0;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-input:focus,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-fw-input:focus,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-fw-input:focus {
  border-color: #6699CC;
  box-shadow: 0 0 0 2px rgba(102,153,204,0.3);
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-btn {
  background: #2a2a3e;
  border-color: #444;
  color: #ccc;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-btn:hover {
  background: #353550;
  border-color: #666;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn-active,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn-active,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-btn-active {
  background: #2a3a5e;
  border-color: #6699CC;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-col-menu,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-col-menu,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-col-menu {
  background: #2a2a3e;
  border-color: #444;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-col-option:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-col-option:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-col-option:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-highlight,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-highlight,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-highlight {
  background-color: #5C4A1E;
  color: #FFE082;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-page-btn {
  background: #2a2a3e;
  border-color: #444;
  color: #ccc;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn:hover,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn:hover,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-page-btn:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn.active,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-page-btn.active,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-page-btn.active {
  background: #6699CC;
  border-color: #6699CC;
  color: #fff;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-pagination,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-pagination,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-pagination {
  color: #999;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-sort-icon {
  color: #555;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-asc .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-asc .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-sort-asc .gd-tbl-sort-icon,
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-desc .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-sort-desc .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-sort-desc .gd-tbl-sort-icon {
  color: #88bbee;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-tooltip,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-tooltip,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-tooltip {
  background: #e0e0e0;
  color: #1a1a2e;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn-copied,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-btn-copied,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-btn-copied {
  color: #4ade80;
  border-color: #4ade80;
}
/* ── Placeholder rows (stable height) ────────────── */
#gd-tbl-5ea5bc9b .gd-tbl-placeholder-row td {
  border-top: none !important;
  border-bottom: none !important;
  padding: 0 !important;
  height: 0;
  line-height: 0;
  overflow: hidden;
  position: relative;
}
#gd-tbl-5ea5bc9b .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #d0d0d0;
}
#gd-tbl-5ea5bc9b .gd-tbl-empty-msg {
  text-align: center;
  color: #999;
  font-size: 13px;
  font-style: italic;
  padding: 8px 0 4px 0;
  user-select: none;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot {
  background: #555;
}
body.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-empty-msg,
html.quarto-dark #gd-tbl-5ea5bc9b .gd-tbl-empty-msg,
:root[data-bs-theme="dark"] #gd-tbl-5ea5bc9b .gd-tbl-empty-msg {
  color: #777;
}
/* ── Column toggle: responsive icon/text ─────────── */
#gd-tbl-5ea5bc9b .gd-tbl-col-btn-icon {
  display: none;
  line-height: 0;
}
#gd-tbl-5ea5bc9b .gd-tbl-col-btn-icon svg {
  display: block;
}
@media (max-width: 576px) {
  #gd-tbl-5ea5bc9b .gd-tbl-col-btn-text {
    display: none;
  }
  #gd-tbl-5ea5bc9b .gd-tbl-col-btn-icon {
    display: inline-flex;
  }
  #gd-tbl-5ea5bc9b .gd-tbl-col-btn {
    padding: 5px 7px;
    line-height: 0;
  }
}
</style>


<table class="gt_table" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="9" class="gt_heading gt_title gt_font_normal"><div style="padding-top: 0; padding-bottom: 7px;">
<span class="gd-tbl-badge" style="background-color: #0075FF; color: #FFFFFF; border: 1px solid #0075FF; margin-right: 8px;">Polars</span>Rows10Columns8
</div></th>
</tr>
<tr class="gt_heading">
<th colspan="9" class="gt_heading gt_subtitle gt_font_normal">Aalen-Johansen event table</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="times" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

times

<em>f64</em>

</div></th>
<th id="at_risk" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

at_risk

<em>i64</em>

</div></th>
<th id="count_0" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

count_0

<em>i64</em>

</div></th>
<th id="count_1" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

count_1

<em>i64</em>

</div></th>
<th id="count_2" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

count_2

<em>i64</em>

</div></th>
<th id="overall_survival" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

overall_survival

<em>f64</em>

</div></th>
<th id="state_occupancy_probability_1_at_times" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

state_occupancy_probability_1_at_times

<em>f64</em>

</div></th>
<th id="state_occupancy_probability_2_at_times" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

state_occupancy_probability_2_at_times

<em>f64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right gd-tbl-rownum">0</td>
<td class="gt_row gt_right" style="max-width: 55px">4.3</td>
<td class="gt_row gt_right" style="max-width: 71px">10</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.9</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
<td class="gt_row gt_right" style="max-width: 250px">0</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">1</td>
<td class="gt_row gt_right" style="max-width: 55px">9.7</td>
<td class="gt_row gt_right" style="max-width: 71px">9</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.8</td>
<td class="gt_row gt_right" style="max-width: 250px">0.2</td>
<td class="gt_row gt_right" style="max-width: 250px">0</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">2</td>
<td class="gt_row gt_right" style="max-width: 55px">14.2</td>
<td class="gt_row gt_right" style="max-width: 71px">8</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 141px">0.7</td>
<td class="gt_row gt_right" style="max-width: 250px">0.2</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">3</td>
<td class="gt_row gt_right" style="max-width: 55px">18.6</td>
<td class="gt_row gt_right" style="max-width: 71px">7</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.6</td>
<td class="gt_row gt_right" style="max-width: 250px">0.3</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">4</td>
<td class="gt_row gt_right" style="max-width: 55px">24.1</td>
<td class="gt_row gt_right" style="max-width: 71px">6</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.5</td>
<td class="gt_row gt_right" style="max-width: 250px">0.4</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">5</td>
<td class="gt_row gt_right" style="max-width: 55px">31.5</td>
<td class="gt_row gt_right" style="max-width: 71px">5</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.5</td>
<td class="gt_row gt_right" style="max-width: 250px">0.4</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">6</td>
<td class="gt_row gt_right" style="max-width: 55px">34.8</td>
<td class="gt_row gt_right" style="max-width: 71px">4</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.5</td>
<td class="gt_row gt_right" style="max-width: 250px">0.4</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">7</td>
<td class="gt_row gt_right" style="max-width: 55px">39.2</td>
<td class="gt_row gt_right" style="max-width: 71px">3</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0.333333333333</td>
<td class="gt_row gt_right" style="max-width: 250px">0.566666666667</td>
<td class="gt_row gt_right" style="max-width: 250px">0.1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">8</td>
<td class="gt_row gt_right" style="max-width: 55px">46</td>
<td class="gt_row gt_right" style="max-width: 71px">2</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 141px">0.166666666667</td>
<td class="gt_row gt_right" style="max-width: 250px">0.566666666667</td>
<td class="gt_row gt_right" style="max-width: 250px">0.266666666667</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">9</td>
<td class="gt_row gt_right" style="max-width: 55px">49.9</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 71px">1</td>
<td class="gt_row gt_right" style="max-width: 71px">0</td>
<td class="gt_row gt_right" style="max-width: 141px">0</td>
<td class="gt_row gt_right" style="max-width: 250px">0.733333333333</td>
<td class="gt_row gt_right" style="max-width: 250px">0.266666666667</td>
</tr>
</tbody>
</table>


The risk set decreases as observations experience an event or are censored. The two cumulative state-occupancy columns record the estimated probability of having entered each absorbing state by each observed time.


# Predict at useful horizons


``` python
horizons = pl.Series("times", [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
estimates = predict_aj_estimates(event_table, horizons)
estimates_with_sum = estimates.with_columns(
    pl.sum_horizontal(
        "state_occupancy_probability_0",
        "state_occupancy_probability_1",
        "state_occupancy_probability_2",
    ).alias("probability_sum")
)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;600&display=swap');
#gd-tbl-72d0e46f .gt_table {
  display: table;
  border-collapse: collapse;
  table-layout: fixed;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 14px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
}
#gd-tbl-72d0e46f .gt_heading {
  background-color: #FFFFFF;
  text-align: left;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-right-style: none;
}
#gd-tbl-72d0e46f .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}
#gd-tbl-72d0e46f .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}
#gd-tbl-72d0e46f .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}
#gd-tbl-72d0e46f .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-right-style: none;
}
#gd-tbl-72d0e46f .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: solid;
  border-left-width: 1px;
  border-left-color: #F2F2F2;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #F2F2F2;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}
#gd-tbl-72d0e46f .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
}
#gd-tbl-72d0e46f .gt_row {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #E9E9E9;
  border-left-style: solid;
  border-left-width: 1px;
  border-left-color: #E9E9E9;
  border-right-style: solid;
  border-right-width: 1px;
  border-right-color: #E9E9E9;
  vertical-align: middle;
  overflow-x: hidden;
}
#gd-tbl-72d0e46f .gt_left { text-align: left; }
#gd-tbl-72d0e46f .gt_center { text-align: center; }
#gd-tbl-72d0e46f .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
#gd-tbl-72d0e46f .gt_font_normal { font-weight: normal; }
#gd-tbl-72d0e46f .gt_font_bold { font-weight: bold; }
#gd-tbl-72d0e46f .gt_font_italic { font-style: italic; }
#gd-tbl-72d0e46f .gt_striped { background-color: rgba(128,128,128,0.05); }
#gd-tbl-72d0e46f .gt_from_md > :first-child { margin-top: 0; }
#gd-tbl-72d0e46f .gt_from_md > :last-child { margin-bottom: 0; }
/* Data cell font */
#gd-tbl-72d0e46f .gt_table_body .gt_row {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #333333;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 14px;
}
/* Column label font */
#gd-tbl-72d0e46f .gt_col_heading {
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
  color: #333333;
}
/* Row number styling */
#gd-tbl-72d0e46f .gd-tbl-rownum {
  color: gray;
  font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 10px;
  border-right: 2px solid rgba(102, 153, 204, 0.5) !important;
  text-align: right;
  padding-right: 6px;
  white-space: nowrap;
}
/* Head/tail divider */
#gd-tbl-72d0e46f tr.gd-tbl-divider td,
#gd-tbl-72d0e46f tr.gd-tbl-divider th {
  border-bottom: 2px solid rgba(102, 153, 204, 0.5);
}
/* Missing values */
#gd-tbl-72d0e46f .gd-tbl-missing {
  color: #B22222 !important;
  background-color: rgba(255, 193, 193, 0.35);
}
/* Column label sub-elements */
#gd-tbl-72d0e46f .gd-tbl-colname {
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  padding-bottom: 0;
  margin-bottom: 0;
}
#gd-tbl-72d0e46f .gd-tbl-dtype {
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow: hidden;
  padding-top: 0;
  margin-top: 0;
  color: #666666;
}
/* Header badge styling */
#gd-tbl-72d0e46f .gd-tbl-badge {
  display: inline-block;
  padding: 2px 10px;
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 10px;
  font-weight: bold;
  text-transform: uppercase;
  position: inherit;
}
#gd-tbl-72d0e46f .gd-tbl-badge-rows-label {
  background-color: #eecbff;
  color: #333333;
  border: 1px solid #eecbff;
  margin-left: 5px;
}
#gd-tbl-72d0e46f .gd-tbl-badge-rows-value {
  background-color: transparent;
  color: #333333;
  border: 1px solid #eecbff;
  margin-left: -4px;
  margin-right: 3px;
}
#gd-tbl-72d0e46f .gd-tbl-badge-cols-label {
  background-color: #BDE7B4;
  color: #333333;
  border: 1px solid #BDE7B4;
}
#gd-tbl-72d0e46f .gd-tbl-badge-cols-value {
  background-color: transparent;
  color: #333333;
  border: 1px solid #BDE7B4;
  margin-left: -4px;
}
/* Dark mode */
body.quarto-dark #gd-tbl-72d0e46f .gt_table,
html.quarto-dark #gd-tbl-72d0e46f .gt_table,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_table {
  background-color: #1a1a2e;
  color: #e0e0e0;
  border-top-color: #555;
  border-bottom-color: #555;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_heading,
html.quarto-dark #gd-tbl-72d0e46f .gt_heading,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_heading {
  background-color: #1a1a2e;
  border-bottom-color: #1a1a2e;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_title,
html.quarto-dark #gd-tbl-72d0e46f .gt_title,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_title {
  color: #e0e0e0;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_subtitle,
html.quarto-dark #gd-tbl-72d0e46f .gt_subtitle,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_subtitle {
  color: #b0b0b0;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_col_headings,
html.quarto-dark #gd-tbl-72d0e46f .gt_col_headings,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_col_headings {
  border-top-color: #444;
  border-bottom-color: #444;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_col_heading,
html.quarto-dark #gd-tbl-72d0e46f .gt_col_heading,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_col_heading {
  color: #b0b0b0;
  background-color: #1a1a2e;
  border-left-color: #333;
  border-right-color: #333;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_table_body,
html.quarto-dark #gd-tbl-72d0e46f .gt_table_body,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_table_body {
  border-top-color: #444;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_table_body .gt_row,
html.quarto-dark #gd-tbl-72d0e46f .gt_table_body .gt_row,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_table_body .gt_row {
  color: #d0d0d0;
  border-top-color: #4b4b4b;
  border-bottom-color: #4b4b4b;
  border-left-color: #333;
  border-right-color: #333;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-rownum,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-rownum,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-rownum {
  color: #888;
  border-right-color: rgba(102, 153, 204, 0.4) !important;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-missing,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-missing,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-missing {
  color: #ff6b6b !important;
  background-color: rgba(60, 17, 24, 0.2);
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-dtype,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-dtype,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-dtype {
  color: #888;
}
body.quarto-dark #gd-tbl-72d0e46f .gt_bottom_border,
html.quarto-dark #gd-tbl-72d0e46f .gt_bottom_border,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gt_bottom_border {
  border-bottom-color: #555;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-rows-label,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-rows-label,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-badge-rows-label {
  background-color: #3d2a4d;
  border-color: #3d2a4d;
  color: #e0c0ff;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-rows-value,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-rows-value,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-badge-rows-value {
  border-color: #3d2a4d;
  color: #d0b0e0;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-cols-label,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-cols-label,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-badge-cols-label {
  background-color: #2a4d2a;
  border-color: #2a4d2a;
  color: #b0e0b0;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-cols-value,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-badge-cols-value,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-badge-cols-value {
  border-color: #2a4d2a;
  color: #a0d0a0;
}
</style> <style>
/* ── Table scroll wrapper ────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-scroll {
  overflow-x: auto;
  overflow-y: hidden;
  width: 100%;
}
/* ── Toolbar ─────────────────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px 0;
  align-items: center;
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  font-size: 13px;
}
/* ── Filter bar ──────────────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-filter-bar {
  flex: 1 1 200px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  min-height: 30px;
  padding: 3px 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  position: relative;
}
#gd-tbl-72d0e46f .gd-tbl-filter-tokens {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}
#gd-tbl-72d0e46f .gd-tbl-filter-token {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  padding: 2px 4px 2px 8px;
  background: #e8f0fe;
  border: 1px solid #c4d9f2;
  border-radius: 12px;
  font-size: 11px;
  color: #1a3a5c;
  white-space: nowrap;
  max-width: 260px;
  line-height: 1.4;
}
#gd-tbl-72d0e46f .gd-tbl-filter-token-text {
  overflow: hidden;
  text-overflow: ellipsis;
}
#gd-tbl-72d0e46f .gd-tbl-filter-token-x {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  background: #d0dfef;
  color: #4a6a8a;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 50%;
  padding: 0;
  padding-bottom: 2px;
  line-height: 1;
  flex-shrink: 0;
  transition: background 0.1s, color 0.1s;
}
#gd-tbl-72d0e46f .gd-tbl-filter-token-x:hover {
  background: #a0bdd8;
  color: #1a3a5c;
}
#gd-tbl-72d0e46f .gd-tbl-filter-token-case {
  font-size: 9px;
  font-weight: 700;
  color: #4477aa;
  border: 1px solid #a0bdd8;
  border-radius: 3px;
  padding: 0 3px;
  line-height: 1.4;
  flex-shrink: 0;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
#gd-tbl-72d0e46f .gd-tbl-filter-add {
  flex-shrink: 0;
  border: none;
  background: none;
  padding: 3px;
  color: #6699CC;
}
#gd-tbl-72d0e46f .gd-tbl-filter-add:hover {
  background: #eef3fb;
  border-radius: 3px;
}
#gd-tbl-72d0e46f .gd-tbl-filter-hint {
  color: #b0b0b0;
  font-size: 12px;
  font-style: italic;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding-left: 4px;
  user-select: none;
  pointer-events: none;
}
#gd-tbl-72d0e46f .gd-tbl-filter-hint svg {
  flex-shrink: 0;
  stroke: #b0b0b0;
}
/* ── Filter wizard dropdown ──────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-filter-wizard {
  position: absolute;
  top: calc(100% + 2px);
  left: 0;
  z-index: 200;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 6px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  padding: 8px 0;
  min-width: 200px;
  max-width: 320px;
  max-height: 300px;
  overflow-y: auto;
  font-size: 12px;
}
#gd-tbl-72d0e46f .gd-tbl-fw-label {
  display: block;
  padding: 4px 12px 4px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #888;
  font-weight: 600;
}
#gd-tbl-72d0e46f .gd-tbl-fw-options {
  display: flex;
  flex-direction: column;
}
#gd-tbl-72d0e46f .gd-tbl-fw-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5px 12px;
  border: none;
  background: none;
  text-align: left;
  font-size: 12px;
  font-family: inherit;
  color: #333;
  cursor: pointer;
  transition: background 0.1s;
}
#gd-tbl-72d0e46f .gd-tbl-fw-option:hover {
  background: #f0f4fb;
}
#gd-tbl-72d0e46f .gd-tbl-fw-dtype {
  font-size: 9px;
  color: #999;
  background: #f0f0f0;
  padding: 1px 5px;
  border-radius: 3px;
  margin-left: 8px;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
}
#gd-tbl-72d0e46f .gd-tbl-fw-input {
  margin: 4px 12px;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
  font-family: inherit;
  background: #fff;
  color: #333;
  outline: none;
  width: calc(100% - 24px);
  box-sizing: border-box;
}
#gd-tbl-72d0e46f .gd-tbl-fw-input:focus {
  border-color: #6699CC;
  box-shadow: 0 0 0 2px rgba(102,153,204,0.2);
}
#gd-tbl-72d0e46f .gd-tbl-fw-between {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 4px;
}
#gd-tbl-72d0e46f .gd-tbl-fw-between .gd-tbl-fw-input {
  flex: 1;
  margin: 4px;
  min-width: 60px;
}
#gd-tbl-72d0e46f .gd-tbl-fw-sep {
  font-size: 11px;
  color: #888;
  flex-shrink: 0;
}
#gd-tbl-72d0e46f .gd-tbl-fw-commit {
  margin: 4px 12px 6px;
  font-size: 11px;
  padding: 4px 14px;
}
#gd-tbl-72d0e46f .gd-tbl-fw-input-row {
  display: flex;
  align-items: center;
  gap: 0;
  padding: 0 8px;
}
#gd-tbl-72d0e46f .gd-tbl-fw-input-row .gd-tbl-fw-input {
  flex: 1;
  margin: 4px 0;
  width: auto;
}
#gd-tbl-72d0e46f .gd-tbl-fw-case {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 26px;
  margin-left: 4px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f8f8f8;
  color: #999;
  font-size: 11px;
  font-weight: 700;
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  cursor: pointer;
  flex-shrink: 0;
  transition: all 0.15s;
}
#gd-tbl-72d0e46f .gd-tbl-fw-case:hover {
  border-color: #999;
  color: #666;
}
#gd-tbl-72d0e46f .gd-tbl-fw-case.active {
  background: #e0edff;
  border-color: #6699CC;
  color: #336699;
}
#gd-tbl-72d0e46f .gd-tbl-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 5px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f8f8f8;
  color: #333;
  font-size: 12px;
  font-weight: 500;
  font-family: inherit;
  line-height: 14px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  white-space: nowrap;
}
#gd-tbl-72d0e46f .gd-tbl-btn:hover {
  background: #eee;
  border-color: #aaa;
}
#gd-tbl-72d0e46f .gd-tbl-btn:focus-visible {
  outline: 2px solid #6699CC;
  outline-offset: 1px;
}
#gd-tbl-72d0e46f .gd-tbl-btn-active {
  background: #e0edff;
  border-color: #6699CC;
}
#gd-tbl-72d0e46f .gd-tbl-btn-icon {
  padding: 5px 7px;
  line-height: 0;
}
#gd-tbl-72d0e46f .gd-tbl-btn-icon svg {
  display: block;
}
/* Copy-success green checkmark state */
#gd-tbl-72d0e46f .gd-tbl-btn-copied {
  color: #198754;
  border-color: #198754;
}
/* ── Button wrapper + tooltip ────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-btn-wrap {
  position: relative;
  display: inline-block;
}
#gd-tbl-72d0e46f .gd-tbl-tooltip {
  visibility: hidden;
  opacity: 0;
  position: absolute;
  top: calc(100% + 4px);
  left: 50%;
  transform: translateX(-50%);
  padding: 3px 8px;
  background: #333;
  color: #fff;
  border-radius: 3px;
  font-size: 11px;
  white-space: nowrap;
  pointer-events: none;
  transition: opacity 0.15s;
  z-index: 100;
}
/* Keep tooltip from overflowing right edge */
#gd-tbl-72d0e46f .gd-tbl-btn-wrap:last-child .gd-tbl-tooltip {
  left: auto;
  right: 0;
  transform: none;
}
#gd-tbl-72d0e46f .gd-tbl-btn-wrap:hover .gd-tbl-tooltip {
  visibility: visible;
  opacity: 1;
}
/* ── Column toggle dropdown ──────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-col-wrap {
  position: relative;
  display: inline-block;
}
#gd-tbl-72d0e46f .gd-tbl-col-wrap .gd-tbl-tooltip {
  left: auto;
  right: 0;
  transform: none;
}
#gd-tbl-72d0e46f .gd-tbl-col-menu {
  display: none;
  position: absolute;
  top: 100%;
  right: 0;
  z-index: 10;
  min-width: 180px;
  max-height: 300px;
  overflow-y: auto;
  margin-top: 4px;
  padding: 6px 0;
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
#gd-tbl-72d0e46f .gd-tbl-col-menu.open {
  display: block;
}
#gd-tbl-72d0e46f .gd-tbl-col-option {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  cursor: pointer;
  font-size: 12px;
  user-select: none;
}
#gd-tbl-72d0e46f .gd-tbl-col-option:hover {
  background: #f0f0f0;
}
/* ── Sort indicators ─────────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-sortable {
  cursor: pointer;
  user-select: none;
  position: relative;
}
#gd-tbl-72d0e46f .gd-tbl-sort-icon {
  display: inline-block;
  width: 10px;
  height: 14px;
  margin-left: 4px;
  color: #bbb;
  vertical-align: middle;
}
#gd-tbl-72d0e46f .gd-tbl-sort-icon svg {
  display: block;
  width: 10px;
  height: 14px;
  fill: currentColor;
}
#gd-tbl-72d0e46f .gd-tbl-sort-asc .gd-tbl-sort-icon,
#gd-tbl-72d0e46f .gd-tbl-sort-desc .gd-tbl-sort-icon {
  color: #6699CC;
}
/* ── Search highlight ────────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-highlight {
  background-color: #FFEEBA;
  border-radius: 2px;
  padding: 0 1px;
}
/* ── Pagination ──────────────────────────────────── */
#gd-tbl-72d0e46f .gd-tbl-pagination {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 0;
  font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
  font-size: 12px;
  color: #666;
}
#gd-tbl-72d0e46f .gd-tbl-page-info {
  white-space: nowrap;
}
#gd-tbl-72d0e46f .gd-tbl-page-nav {
  display: flex;
  gap: 2px;
  align-items: center;
}
#gd-tbl-72d0e46f .gd-tbl-page-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  padding: 0 6px;
  border: 1px solid #ddd;
  border-radius: 3px;
  background: #fff;
  color: #333;
  cursor: pointer;
  font-size: 12px;
  font-family: inherit;
  transition: background 0.1s;
}
#gd-tbl-72d0e46f .gd-tbl-page-btn:hover {
  background: #f0f0f0;
}
#gd-tbl-72d0e46f .gd-tbl-page-btn.active {
  background: #6699CC;
  color: #fff;
  border-color: #6699CC;
}
#gd-tbl-72d0e46f .gd-tbl-page-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
#gd-tbl-72d0e46f .gd-tbl-page-ellipsis {
  padding: 0 4px;
  color: #999;
}
/* ── Dark mode ───────────────────────────────────── */
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-bar,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-bar,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-bar {
  background-color: #2a2a3e;
  border-color: #444;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-token {
  background: #2d3a50;
  border-color: #3d5070;
  color: #b0ccee;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token-x:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token-x:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-token-x:hover {
  background: #3d5070;
  color: #e0e8f0;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token-case,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-token-case,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-token-case {
  color: #88bbee;
  border-color: #4d6888;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-case {
  background: #2a2a3e;
  border-color: #555;
  color: #888;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-case:hover {
  border-color: #888;
  color: #bbb;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case.active,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-case.active,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-case.active {
  background: #2d3a50;
  border-color: #6699CC;
  color: #88bbee;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-add,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-add,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-add {
  color: #88bbee;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-add:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-add:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-add:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-hint,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-hint,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-hint {
  color: #666;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-hint svg,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-hint svg,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-hint svg {
  stroke: #666;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-wizard,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-filter-wizard,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-filter-wizard {
  background: #1e1e32;
  border-color: #444;
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-option,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-option,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-option {
  color: #ddd;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-option:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-option:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-option:hover {
  background: #2a2a44;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-dtype,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-dtype,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-dtype {
  background: #333;
  color: #aaa;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-input,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-input,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-input {
  background: #2a2a3e;
  border-color: #555;
  color: #e0e0e0;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-input:focus,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-fw-input:focus,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-fw-input:focus {
  border-color: #6699CC;
  box-shadow: 0 0 0 2px rgba(102,153,204,0.3);
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-btn {
  background: #2a2a3e;
  border-color: #444;
  color: #ccc;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-btn:hover {
  background: #353550;
  border-color: #666;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn-active,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn-active,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-btn-active {
  background: #2a3a5e;
  border-color: #6699CC;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-col-menu,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-col-menu,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-col-menu {
  background: #2a2a3e;
  border-color: #444;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-col-option:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-col-option:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-col-option:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-highlight,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-highlight,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-highlight {
  background-color: #5C4A1E;
  color: #FFE082;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-page-btn {
  background: #2a2a3e;
  border-color: #444;
  color: #ccc;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn:hover,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn:hover,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-page-btn:hover {
  background: #353550;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn.active,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-page-btn.active,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-page-btn.active {
  background: #6699CC;
  border-color: #6699CC;
  color: #fff;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-pagination,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-pagination,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-pagination {
  color: #999;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-sort-icon {
  color: #555;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-asc .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-asc .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-sort-asc .gd-tbl-sort-icon,
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-desc .gd-tbl-sort-icon,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-sort-desc .gd-tbl-sort-icon,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-sort-desc .gd-tbl-sort-icon {
  color: #88bbee;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-tooltip,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-tooltip,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-tooltip {
  background: #e0e0e0;
  color: #1a1a2e;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn-copied,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-btn-copied,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-btn-copied {
  color: #4ade80;
  border-color: #4ade80;
}
/* ── Placeholder rows (stable height) ────────────── */
#gd-tbl-72d0e46f .gd-tbl-placeholder-row td {
  border-top: none !important;
  border-bottom: none !important;
  padding: 0 !important;
  height: 0;
  line-height: 0;
  overflow: hidden;
  position: relative;
}
#gd-tbl-72d0e46f .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #d0d0d0;
}
#gd-tbl-72d0e46f .gd-tbl-empty-msg {
  text-align: center;
  color: #999;
  font-size: 13px;
  font-style: italic;
  padding: 8px 0 4px 0;
  user-select: none;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-placeholder-row td .gd-tbl-placeholder-dot {
  background: #555;
}
body.quarto-dark #gd-tbl-72d0e46f .gd-tbl-empty-msg,
html.quarto-dark #gd-tbl-72d0e46f .gd-tbl-empty-msg,
:root[data-bs-theme="dark"] #gd-tbl-72d0e46f .gd-tbl-empty-msg {
  color: #777;
}
/* ── Column toggle: responsive icon/text ─────────── */
#gd-tbl-72d0e46f .gd-tbl-col-btn-icon {
  display: none;
  line-height: 0;
}
#gd-tbl-72d0e46f .gd-tbl-col-btn-icon svg {
  display: block;
}
@media (max-width: 576px) {
  #gd-tbl-72d0e46f .gd-tbl-col-btn-text {
    display: none;
  }
  #gd-tbl-72d0e46f .gd-tbl-col-btn-icon {
    display: inline-flex;
  }
  #gd-tbl-72d0e46f .gd-tbl-col-btn {
    padding: 5px 7px;
    line-height: 0;
  }
}
</style>


<table class="gt_table" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="7" class="gt_heading gt_title gt_font_normal"><div style="padding-top: 0; padding-bottom: 7px;">
<span class="gd-tbl-badge" style="background-color: #0075FF; color: #FFFFFF; border: 1px solid #0075FF; margin-right: 8px;">Polars</span>Rows6Columns6
</div></th>
</tr>
<tr class="gt_heading">
<th colspan="7" class="gt_heading gt_subtitle gt_font_normal">State-occupancy estimates at requested horizons</th>
</tr>
<tr class="gt_col_headings">
<th class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="times" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

times

<em>f64</em>

</div></th>
<th id="state_occupancy_probability_0" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

state_occupancy_probability_0

<em>f64</em>

</div></th>
<th id="state_occupancy_probability_1" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

state_occupancy_probability_1

<em>f64</em>

</div></th>
<th id="state_occupancy_probability_2" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

state_occupancy_probability_2

<em>f64</em>

</div></th>
<th id="estimate_origin" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"><div>

estimate_origin

<em>enum</em>

</div></th>
<th id="probability_sum" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"><div>

probability_sum

<em>f64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right gd-tbl-rownum">0</td>
<td class="gt_row gt_right" style="max-width: 55px">0</td>
<td class="gt_row gt_right" style="max-width: 242px">1</td>
<td class="gt_row gt_right" style="max-width: 242px">0</td>
<td class="gt_row gt_right" style="max-width: 242px">0</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">1</td>
<td class="gt_row gt_right" style="max-width: 55px">10</td>
<td class="gt_row gt_right" style="max-width: 242px">0.8</td>
<td class="gt_row gt_right" style="max-width: 242px">0.2</td>
<td class="gt_row gt_right" style="max-width: 242px">0</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">2</td>
<td class="gt_row gt_right" style="max-width: 55px">20</td>
<td class="gt_row gt_right" style="max-width: 242px">0.6</td>
<td class="gt_row gt_right" style="max-width: 242px">0.3</td>
<td class="gt_row gt_right" style="max-width: 242px">0.1</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">3</td>
<td class="gt_row gt_right" style="max-width: 55px">30</td>
<td class="gt_row gt_right" style="max-width: 242px">0.5</td>
<td class="gt_row gt_right" style="max-width: 242px">0.4</td>
<td class="gt_row gt_right" style="max-width: 242px">0.1</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">4</td>
<td class="gt_row gt_right" style="max-width: 55px">40</td>
<td class="gt_row gt_right" style="max-width: 242px">0.333333333333</td>
<td class="gt_row gt_right" style="max-width: 242px">0.566666666667</td>
<td class="gt_row gt_right" style="max-width: 242px">0.1</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
<tr>
<td class="gt_row gt_right gd-tbl-rownum">5</td>
<td class="gt_row gt_right" style="max-width: 55px">50</td>
<td class="gt_row gt_right" style="max-width: 242px">-5.55111512313e-17</td>
<td class="gt_row gt_right" style="max-width: 242px">0.733333333333</td>
<td class="gt_row gt_right" style="max-width: 242px">0.266666666667</td>
<td class="gt_row gt_left" style="max-width: 153px">fixed_time_horizons</td>
<td class="gt_row gt_right" style="max-width: 133px">1</td>
</tr>
</tbody>
</table>


Every row sums to one, up to floating-point precision. State 0 means no absorbing event has occurred by the horizon; states 1 and 2 are the cumulative probabilities of the event of interest and competing event.

> **Tip: Tip**
>
> To model a simple single-event analysis, use the same workflow with only codes `0` and `1`. The state-2 probability remains zero.
