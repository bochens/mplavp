# MPL Aerosol Vertical Profile Retrieval (AVP)

This repository contains Python code for retrieving aerosol vertical profiles using data from the Micropulse Lidar (MPL). Developed by Sarah D. Brooks group at Texas A&M University. 

This code provides routines for:

- Processing raw MPL data
- Performing Fernald inversion
- Identifying cloud and layering structure
- Ingesting radiosonde data
- Ingesting ground-based aerosol measurements (e.g., size distribution, CCN, INP)
- Plotting relevant diagnostic and retrieval figures

This code was initially developed for the *Tracking Aerosol Convection Interactions Experiment* (TRACER), funded by the U.S. Department of Energy.  
Included data in this repository is for example only.

## Publications

If you use this code or methodology in your work, please consider citing the following publications:

- Chen, B., Thompson, S. A., Matthews, B. H., Sharma, M., Li, R., Nowotarski, C. J., ... & Brooks, S. D. (2024). A New Technique to Retrieve Aerosol Vertical Profiles Using Micropulse Lidar and Ground-based Aerosol Measurements. EGUsphere, 2024, 1-33.

## Repository Structure

- `LICENSE` – Project license
- `README.md` – Project overview and documentation
- `dev/`
  - `find_layers.py` – Layer identification routines
  - `inversempl.py` – Fernald inversion implementation
  - `kappa_kohler_theory.py` – κ-Köhler theory utilities
  - `plotmpl.py` – Plotting functions
  - `pympl.py` – MPL data loading and preprocessing
  - `retrieval_aux.py` – Auxiliary functions
  - `tracer_avp_processing.py` – Main TRACER pipeline