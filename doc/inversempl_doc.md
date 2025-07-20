# InverseMPL

## Overview
Utility functions for aerosol/molecular lidar inversion and signal smoothing. Implements several classic algorithms (Fernald, Klett, Sasano) plus wavelet–based denoising helpers.

---
## Modules & Globals
| symbol | description |
|--------|-------------|
| `molecular_lidar_ratio` | Constant  \(8\pi/3\)  used for Rayleigh lidar‑ratio (≈ 8.377). |
| `US_standard_pressure`, `US_standard_temperature` | Interpolators for U.S. Standard Atmosphere (pressure [Pa], temperature [K]). |

---
## Function Reference
Each entry follows NumPy style: short summary → **Parameters** → **Returns**.

### `calculate_backscatter_coefficient(LAM, pressure, temperature)`
Compute molecular volume backscatter coefficient.

**Parameters**
- `LAM` (`float | np.ndarray`): Wavelength **µm**.
- `pressure` (`float | np.ndarray`): Pressure **Pa** (same shape as `temperature`).
- `temperature` (`float | np.ndarray`): Temperature **K**.

**Returns**
- `np.ndarray[float]`: Volume backscatter \(\beta\) **km⁻¹ sr⁻¹**.

---
### `rayleigh_bcksca_coeff(LAM, height, pressure_interpolater=US_standard_pressure, temperature_interpolater=US_standard_temperature)`
Rayleigh backscatter coefficient at a given altitude using the supplied atmosphere.

**Parameters**
- `LAM` (`float`): Wavelength **µm**.
- `height` (`float | np.ndarray`): Altitude **km**.
- `pressure_interpolater` (`Callable`): Returns pressure **Pa** for height.
- `temperature_interpolater` (`Callable`): Returns temperature **K** for height.

**Returns**
- `np.ndarray[float]`: Molecular backscatter **km⁻¹ sr⁻¹**.

---
### `Fernald_inversion_inwards(nrb, mpl_range, beta2, S1, S2=8π/3, calibration_beta1=0, calibration_range=20, LAM=0.532)`
Two‑component Fernald inversion **toward the instrument**.

**Parameters**
- `nrb` (`np.ndarray[float]`): Attenuated backscatter `(t?, r)` counts km² µs⁻¹ µJ⁻¹.
- `mpl_range` (`np.ndarray[float]`): Range grid **km**.
- `beta2` (`np.ndarray[float]`): Molecular backscatter profile.
- `S1` (`float`): Aerosol lidar ratio (sr).
- `S2` (`float`, default molecular ratio): Molecular lidar ratio (sr).
- `calibration_beta1` (`float | np.ndarray`): Aerosol backscatter at `calibration_range`.
- `calibration_range` (`float`): Calibration altitude **km**.
- `LAM` (`float`): Wavelength **µm** (informational).

**Returns**
- `beta1` (`np.ndarray[float]`): Retrieved aerosol backscatter `(t?, r)` **km⁻¹ sr⁻¹**.
- `beta2_sel` (`np.ndarray[float]`): Molecular backscatter subset.
- `inv_range` (`np.ndarray[float]`): Range grid used.

---
### `Sasano_inversion_inwards(...)`
Variable‑lidar‑ratio inwards inversion (Sasano et al.).

**Returns**
- `beta1` (`np.ndarray[float]`): Retrieved aerosol backscatter `(t?, r)` **km⁻¹ sr⁻¹**.
- `beta2_sel` (`np.ndarray[float]`): Molecular backscatter subset.
- `inv_range` (`np.ndarray[float]`): Range grid used.

---
### `Fernald_inversion_outwards(...)`
Outward version of the two‑component Fernald solution.

**Parameters & Returns**: As for `Fernald_inversion_inwards`, but integration proceeds away from instrument.

---
### `Klett_inversion_inwards(nrb, mpl_range, S1, calibration_beta=0, calibration_range=20)`
Single‑component Klett inversion toward instrument.

**Parameters**
- `nrb` (`np.ndarray`): NRB.
- `mpl_range` (`np.ndarray`): Range **km**.
- `S1` (`float`): Aerosol lidar ratio.
- `calibration_beta` (`float | np.ndarray`): Calibration backscatter.
- `calibration_range` (`float`): Calibration altitude **km**.

**Returns**
- `beta1` (`np.ndarray`): Aerosol backscatter.
- `inv_range` (`np.ndarray`): Range used.

---
### `iterative_aod_Fernald_inversion_inwards(...)`
Iteratively adjusts aerosol lidar ratio to match a given AOD.

**Returns**
- `beta1` (`np.ndarray`), `beta2_sel` (`np.ndarray`), `inv_range` (`np.ndarray`), `converged_S1` (`float`).

---
### Wavelet Denoising & Smoothing
#### `signal_smoothing(...)`
Apply neighbourhood‑block wavelet smoothing (1‑D/2‑D).

#### `signal_smoothing_snr_selection(...)`
Wavelet smoothing weighted by SNR profile.

*(Helper functions `_signal_smoothing`, `_signal_smoothing_snr_selection`, `plot_dwt`, `neigh_block` are internal utilities.)*

---
### Slope‑based Retrievals
#### `slope_inversion(...)`
Klett‑style slope method to retrieve extinction/backscatter using linear fit to log‑NRB.

#### `iterative_slope_inversion(...)`
Iterative variant converging lidar ratio via slope constraints.

