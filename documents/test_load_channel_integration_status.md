# test_load_channel Integration Status

**Goal:** Consolidate `test_load_channel.py`, `test_effective_area_loader.py`, and `test_load_channel_effective_area.py` into a single `test_load_channel.py`.

---

## 1. Duplicate Files (REMOVE) — DONE

| File | Status |
|------|--------|
| `test_effective_area_loader.py` | **Deleted** |
| `test_load_channel_effective_area.py` | **Deleted** |

**Done:** Both removed. Unique content merged into `test_load_channel.py`.

---

## 2. Overlap: load_effective_area_file — DONE

| Test | Kept in test_load_channel | Removed (EA loader) |
|------|---------------------------|--------------------|
| Success (basic parse) | `test_load_effective_area_file_success` | `test_ok_parses_pixel_scale_and_first_last_columns` |
| Missing file | `test_load_effective_area_file_missing_file_raises` | `test_missing_file_raises_valueerror` |
| Missing pixel scale header | `test_load_effective_area_file_missing_pixel_scale_raises` | `test_missing_pixel_scale_header_raises_valueerror` |
| No numeric data | `test_load_effective_area_file_no_numeric_data_raises` | `test_no_numeric_rows_raises_valueerror` |

**Done:** Kept test_load_channel versions. EA-loader duplicates removed with file deletion.

---

## 3. Unique in test_effective_area_loader / test_load_channel_effective_area — DONE

These 8 tests were added to `test_load_channel.py`:

| Test (in test_load_channel) | Purpose |
|----------------------------|---------|
| `test_load_effective_area_file_invalid_pixel_scale_raises` | Non-numeric pixel scale in header |
| `test_load_effective_area_file_one_column_table_raises` | 1D table (ndim==1 guard) |
| `test_load_effective_area_file_one_row_two_columns_raises` | Single row → 1D from np.loadtxt |
| `test_load_effective_area_file_header_lines_after_pixel_scale_ok` | Extra comments after pixel scale |
| `test_load_effective_area_file_extra_columns_first_and_last_used` | Middle columns ignored, first/last used |
| `test_load_effective_area_file_leading_trailing_whitespace_ok` | Tabs/whitespace in numeric block |
| `test_load_effective_area_file_blank_lines_inside_numeric_block_ok` | Blank lines between rows |
| `test_load_effective_area_file_malformed_numeric_row_raises` | BAD value in a row → parse failure |

**Done:** All 8 added. Use `_REPO_ROOT`, `_write` helper.

---

## 4. Unique in test_effective_area_loader: load_spread_profile_file — DONE

| Test | test_load_channel | EA loader (removed) |
|------|-------------------|--------------------|
| Empty filename → None | `test_load_channel_config_calls_spread_loader_with_empty_filename_and_sets_none` | — |
| Success parse | `test_load_spread_profile_file_success` | — |
| Missing file | `test_load_spread_profile_file_missing_file_raises` | — |
| Missing pixels header | `test_load_spread_profile_file_missing_pixels_header_raises` | Added |
| Header/weight col mismatch | `test_load_spread_profile_file_header_count_mismatch_raises` | Added |
| Whitespace ok | `test_load_spread_profile_file_leading_trailing_whitespace_ok` | Added |

**Done:** Kept overlap tests. Added 3 unique spread tests to `test_load_channel.py`.

---

## 5. Brittle / Implementation-Detail Tests — DONE

| Test | Change |
|------|--------|
| `test_load_channel_config_raises_if_wavelength_length_does_not_match_x_pixels` | **Softened:** Assert key semantics (channel, file, "len(wavelength)", "x_pixels") instead of exact message format. |
| `test_load_channel_config_calls_ea_loader_with_effective_area_file` | Keep: integration test for loader call. |
| `test_load_channel_config_ir_returns_photometry_channel_without_loading_ea` | Keep: valid behavior test. |

**Done:** No hardcoded debug/error message strings. Tests assert semantic content only.

---

## 6. Monkeypatch Inconsistency — DONE

**Done:** Single file `test_load_channel.py` uses `monkeypatch.setattr(_REPO_ROOT, lambda: tmp_path)` consistently. `_REPO_ROOT = "loaders.load_channel.get_repo_root"`. EA loader files removed.

---

## 7. Helper Consolidation

| test_load_channel | test_effective_area_loader |
|-------------------|---------------------------|
| `_write_cfg(path, **kwargs)` | — |
| `_write_ea_file(path, pixel_scale, rows)` | — |
| `_write_spread_file(path, wavelengths, num_rows)` | — |
| — | `_write(path, text)` — generic |

**Action:** Add `_write(path, text)` for inline file content in edge-case tests. Keep `_write_ea_file` and `_write_spread_file` for common cases.

---

## 8. Summary — DONE

| Action | Status |
|--------|--------|
| **Delete** | Done: `test_effective_area_loader.py`, `test_load_channel_effective_area.py` |
| **Keep** | All tests in `test_load_channel.py` |
| **Add** | Done: 8 EA loader + 3 spread loader tests |
| **Refactor** | Done: `_REPO_ROOT`, `_write(path, text)` helper |
| **Soften** | Optional: brittle asserts (deferred) |

---

## 9. Final test_load_channel.py Structure (Proposed)

```
# Imports, constants, helpers
# _UserCfgChannels, _write_cfg, _write_ea_file, _write_spread_file, _write, _no_spread

# --- load_channel_config: loader calls and return values ---
# (existing 6 tests)

# --- load_channel_config: _parse_simple_kv and validation ---
# (existing 8 tests)

# --- load_effective_area_file: direct tests ---
# (existing 4) + (8 new from EA loader)

# --- load_spread_profile_file: direct tests ---
# (existing 3) + (3 new from EA loader)

# --- load_channels_config ---
# (existing 1)

# --- IR channel ---
# (existing 1)
```

**Estimated final count:** ~32 tests (22 existing + 11 new, minus any consolidation).
