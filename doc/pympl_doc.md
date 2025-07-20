# PyMPL

## Overview

`PyMPL` is a high‑level container class for reading, correcting, and analysing Micro‑Pulse Lidar (MPL) binary scans together with their calibration data (after‑pulse, overlap, dead‑time).

---

## Constructor

```python
PyMPL(data_input, ap_input, ov_input, dt_input, blind_range=0.1)
```

| parameter    | type                          | description                                                  |
| ------------ | ----------------------------- | ------------------------------------------------------------ |
| `data_input` | `str`, `list[str]`, or `dict` | Path(s) to one or more *.mpl* files **or** a fully parsed data dictionary identical to `self.data_dict`. |
| `ap_input`    | `str` or `dict` | After‑pulse correction file (binary) **or** pre‑parsed dict identical to `self.ap_dict`. |
| `ov_input`    | `str` or `dict` | Overlap correction file (binary) **or** pre‑parsed dict identical to `self.ov_dict`. |
| `dt_input`    | `str` or `dict` | Dead‑time correction file (`.bin` or `.csv`) **or** pre‑parsed dict identical to `self.dt_dict`. |
| `blind_range` | `float`         | Minimum usable range *in kilometres*; range bins closer than this are masked out (default **0.1 km**). |

---

## Attribute

Below is the full set of attributes created at instantiation.  *Interpolated versions* (prefixed with `interpolated_`) are generated only after a call to `interpolate_data()`.

| attribute                                     | dtype                          | description                                                   |
| --------------------------------------------- | ------------------------------ | ------------------------------------------------------------- |
| `datetime`                                    | `np.ndarray["datetime64[ns]"]` | UTC timestamp for each profile.                               |
| `seconds_since_start`                         | `np.ndarray[int]`              | Seconds elapsed since first profile.                          |
| `range`                                       | `np.ndarray[float]`            | Bin‑centre altitude (km), blind‑range filtered.               |
| `range_edges`                                 | `np.ndarray[float]`            | Bin edges (km) for plotting.                                  |
| `bin_resolition`                              | `float`                        | Vertical resolution (km).                                     |
| `laser_energy`                                | `np.ndarray[float]`            | Laser pulse energy (mJ).                                      |
| `temp_detector`                               | `np.ndarray[float]`            | Detector temperature (°C).                                    |
| `temp_telescope`                              | `np.ndarray[float]`            | Telescope temperature (°C).                                   |
| `temp_laser`                                  | `np.ndarray[float]`            | Laser head temperature (°C).                                  |
| `sync_pulses_seen_per_second`                 | `np.ndarray[int] \| None`      | FPGA sync pulses per second (miniMPL only).                   |
| `number_profile`                              | `int`                          | Total number of scans in the object.                          |
| `raw_copol`, `raw_crosspol`                   | `np.ndarray[float]`            | Raw photon counts (counts µs⁻¹) for co‑/cross‑polar channels. |
| `background_copol`, `background_crosspol`     | `np.ndarray[float]`            | Background photon counts (counts µs⁻¹).                       |
| `r2_corrected_copol`, `r2_corrected_crosspol` | `np.ndarray[float]`            | Range‑squared‑corrected signal.                               |
| `nrb_copol`, `nrb_crosspol`, `nrb_unpol`      | `np.ndarray[float]`            | Normalised Relative Backscatter.                              |
| `snr_copol`, `snr_crosspol`                   | `np.ndarray[float]`            | Signal‑to‑noise ratio.                                        |
| `depol_ratio`                                 | `np.ndarray[float]`            | Volume depolarisation ratio.                                  |
| `interpolation_flag`                          | `bool`                         | `True` once any interpolation has been applied.               |

*All of the above gain **`interpolated_…`** counterparts after a call to* `interpolate_data()`.

---

## Method

Below each method is documented in **bullet list** style with data types and return values.

---

#### `deepcopy()`

- **Returns** (`PyMPL`): A fully independent deep copy of the object.

---

#### `interpolation_reset()`

- **Returns** (`None`): Resets `interpolation_flag` and clears all `interpolated_*` attributes.

---

#### `read_files(data_path, ap_path, ov_path, dt_path)`

- `data_path` (`str | list[str]`): Path(s) to one or more `.mpl` files or a directory containing them.
- `ap_path` (`str`): After‑pulse correction binary file.
- `ov_path` (`str`): Overlap correction binary file.
- `dt_path` (`str`): Dead‑time correction file (`.bin` or `.csv`).

**Returns**

- `tuple`: `(data_dict, ap_dict, ov_dict, dt_dict)` – four dictionaries identical in structure to the instance attributes that would be created.

Reads raw data and the three calibration files in one call.

---

#### `read_mpl(file_path)`

- `file_path` (`str | list[str]`): Single file path, directory, or list of paths pointing to `.mpl` files.

**Returns**

- `tuple`: `(data_dict, number_bins)` where `number_bins` is `int`.

Dispatches to single‑ or multi‑file loaders.

---

#### `read_mpl_single_file(file_path)`

- `file_path` (`str`): Path to **one** `.mpl` binary.

**Returns**

- `tuple`: `(data_dict, number_bins)`.

---

#### `read_mpl_multiple_file(file_list)`

- `file_list` (`list[str]`): List of `.mpl` files.

**Returns**

- `tuple`: `(data_dict, number_bins)` with concatenated records.

---

#### `read_afterpulse(file_path, number_bins)`

- `file_path` (`str`): After‑pulse binary file.
- `number_bins` (`int`): Expected bin count check.

**Returns**

- `dict`: Keys include `ap_range`, `ap_copol`, `ap_crosspol`, etc.

---

#### `read_overlap(file_path)`

- `file_path` (`str`): Overlap binary file.

**Returns**

- `dict`: Keys `ol_range`, `ol_overlap`.

---

#### `read_deadtime(file_path)`

- `file_path` (`str`): Dead‑time file (`.bin`/`.csv`).

**Returns**

- `dict`: Keys `dt_coeff`, `dt_coeff_degree`, `dt_number_coeff`.

---

#### `calculate_snr(raw_data, background, background_std_dev)`

- `raw_data` (`np.ndarray[float]`): Shape `(n_profile, n_range)`.
- `background` (`np.ndarray[float]`): Shape `(n_profile,)`.
- `background_std_dev` (`np.ndarray[float]`): Shape `(n_profile,)`.

**Returns**

- `np.ndarray[float]`: Same shape as `raw_data` containing SNR.

---

#### `calculate_dtcf(data)`

- `data` (`np.ndarray[float]`): Photon counts.

**Returns**

- `np.ndarray[float]`: Dead‑time correction factor.

---

#### `calculate_r2_corrected(raw_data, background)`

- `raw_data` (`np.ndarray[float]`): Photon counts.
- `background` (`np.ndarray[float]`): Background counts.

**Returns**

- `np.ndarray[float]`: `(raw − background) × range²`.

---

#### `calculate_nrb(raw_data, background, ap_data, ap_background)`

- `raw_data` (`np.ndarray[float]`): Photon counts.
- `background` (`np.ndarray[float]`): Background counts.
- `ap_data` (`np.ndarray[float]`): After‑pulse profile interpolated to range.
- `ap_background` (`float`): Scalar after‑pulse background.

**Returns**

- `np.ndarray[float]`: Normalised Relative Backscatter.

---

#### `calculate_depol_ratio()`

- **Returns** (`np.ndarray[float]`): Volume depolarisation ratio.

---

#### `select_time(start_time, end_time)`

- `start_time` (`str | datetime | np.datetime64`): Inclusive start.
- `end_time` (same): Inclusive end.

**Returns**

- `PyMPL`: New object containing only scans in range.

---

#### `select_snr(data, snr_data, snr_limit)` *(class method)*

- `data` (`np.ndarray`): Signal to mask.
- `snr_data` (`np.ndarray`): SNR array.
- `snr_limit` (`float`): Threshold below which data are set `NaN`.

**Returns**

- `np.ndarray`: Masked copy of `data`.

---

#### `interpolate_data(time_resolution, start_time=None, end_time=None, gap_seconds=None, mov_avg_win=None)`

- `time_resolution` (`int`): Output step (seconds).
- `start_time` (`optional`): Defaults to first profile.
- `end_time` (`optional`): Defaults to last profile.
- `gap_seconds` (`int | None`): Gap threshold (default `2×time_resolution`).
- `mov_avg_win` (`int | None`): Moving‑average window applied before interpolation.

**Returns**

- `None`: Updates object in‑place; creates all `interpolated_*` arrays.

---

#### `interpolate_single_data(new_time_array, data, time_resolution=None, gap_seconds=None)`

- `new_time_array` (`np.ndarray[datetime64[s]]`): Regular grid.
- `data` (`np.ndarray`): Series to interpolate.
- `time_resolution` (`int | None`): Seconds per step (auto if None).
- `gap_seconds` (`int | None`): Gap threshold.

**Returns**

- `np.ndarray`: Interpolated data.

---

#### `make_time_array(time_resolution, start_time, end_time)`

- `time_resolution` (`int`): Seconds.
- `start_time`, `end_time` (`datetime | np.datetime64 | str`): Bounds.

**Returns**

- `np.ndarray[datetime64[s]]`: Regular grid including `end_time`.

---

#### `movingaverage(values, window, axis=0)`

- `values` (`np.ndarray` 1‑D/2‑D): Input.
- `window` (`int`): Window length.
- `axis` (`int`): 0=row, 1=column.

**Returns**

- `np.ndarray`: Smoothed array same shape as input.

---

#### `_movingaverage(values, window)`

- `values` (`np.ndarray[float]`): 1‑D signal.
- `window` (`int`): Window length.

**Returns**

- `np.ndarray[float]`: Smoothed 1‑D array.

---

#### `write_mpl(output_dir, filename)`

- `output_dir` (`str`): Directory.
- `filename` (`str`): Base name.

**Returns**

- `None`: Writes `<filename>.mpl` to disk.

---

#### `output_netcdf(output_dir, filename)` *(placeholder)*

- `output_dir` (`str`)
- `filename` (`str`)

**Returns**

- `None`: Not yet implemented.

---

#### `get_file_list_by_date_range(mpl_file_folder, date_range, suffix='*.mpl')`

- `mpl_file_folder` (`str`): Directory.
- `date_range` (`np.ndarray[datetime64[D]]`): Daily array.
- `suffix` (`str`): Glob pattern.

**Returns**

- `list[str]`: Matching files.

---

#### `get_file_list_by_start_end_datetime(mpl_file_folder, start_datetime, end_datetime, suffix='*.mpl')`

- `mpl_file_folder` (`str`)
- `start_datetime` (`datetime | np.datetime64`)
- `end_datetime` (`datetime | np.datetime64`)
- `suffix` (`str`)

**Returns**

- `list[str]`: Files spanning the period plus one before and after.

---

#### `get_date_range(start_datetime, end_datetime)`

- `start_datetime` (`datetime | np.datetime64`)
- `end_datetime` (`datetime | np.datetime64`)

**Returns**

- `np.ndarray[datetime64[D]]`: Daily dates inclusive.

