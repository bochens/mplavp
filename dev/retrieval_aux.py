# Standard library imports
import os
import re
import csv
import datetime
import warnings

# Numerical and scientific computing
import numpy as np
import scipy

# NetCDF handling
import netCDF4 as nc  # use only the aliased import

# Plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# AUX FUNCTIONS FOR HANDLING DATA FOR RETRIEVAL

def read_merged_data(date_range, smps_pops_merged_dist_path):
    table_field = None
    merged_sizes = None
    all_merged_datetime = []
    all_merged_ri_n = []
    all_merged_ri_k = []
    all_merged_size_dist = []
    for i, a_date in enumerate(date_range):
        a_date_str = a_date.astype('M8[D]').astype(str)[2:4] + a_date.astype('M8[D]').astype(str)[5:7] + a_date.astype('M8[D]').astype(str)[8:]
        merged_data_set = np.loadtxt(os.path.join(smps_pops_merged_dist_path, a_date_str+'.csv'), dtype=np.string_, delimiter=',')
        table_field = merged_data_set[0,:]
        merged_sizes = table_field[3:].astype(float)
        
        merged_datetime  = merged_data_set[1:, 0].astype('datetime64')
        merged_ri_n      = merged_data_set[1:, 1].astype(float)
        merged_ri_k      = merged_data_set[1:, 2].astype(float)
        merged_size_dist = merged_data_set[1:,3:].astype(float)

        # This code block below will take care of multi day reading of the SMPS+POPS merged data, and fill in between with np.nans
        if i != 0:
            last_datetime   = all_merged_datetime[i-1][-1]
            first_datetime  = merged_datetime[0]
            inbetween       = np.arange(last_datetime + np.timedelta64(1, 'm'), first_datetime, np.timedelta64(1, 'm'))
            inbetween_len  = len(inbetween)
            merged_datetime = np.concatenate((inbetween, merged_datetime))
            merged_ri_n     = np.concatenate((np.full(inbetween_len, np.nan), merged_ri_n))
            merged_ri_k     = np.concatenate((np.full(inbetween_len, np.nan), merged_ri_k))
            merged_size_dist = np.vstack((np.full((inbetween_len, len(merged_sizes)), np.nan), merged_size_dist))

        all_merged_datetime.append(merged_datetime)
        all_merged_ri_n.append(merged_ri_n)
        all_merged_ri_k.append(merged_ri_k)
        all_merged_size_dist.append(merged_size_dist)

    all_merged_datetime = np.concatenate(all_merged_datetime)
    all_merged_ri_n = np.concatenate(all_merged_ri_n)
    all_merged_ri_k = np.concatenate(all_merged_ri_k)
    all_merged_size_dist = np.vstack(all_merged_size_dist)
    
    return merged_sizes, all_merged_datetime, all_merged_ri_n, all_merged_ri_k, all_merged_size_dist

def read_tmatrix_data(folder_path, EPS, wavelength, ri_n, ri_k):
    '''
    Reads the tmatrix data based on EPS, wavelength, ri_n, and ri_k

    EPS: Specify the shape of the spheroid particle.
    '''

    if EPS==1:
        EPS = 1.0000010

    if np.isnan(ri_n) or np.isnan(ri_k):
        return None, None, None, None, None
    
    else:
        # Format the numbers to match the desired format
        EPS_str = "{:.7f}".format(EPS)
        wavelength_str = "{:.3f}".format(wavelength)
        ri_n_str = "{:.3f}".format(ri_n)
        ri_k_str = "{:.3f}".format(ri_k)

        # "/Users/bochen/TAMU/tamu_tracer/T-matrix_results/LAM=0.532 EPS=1.0000010 k=0.001/RANDOMLY ORIENTED OBLATE SPHEROIDS EPS=1.0000010 lambda=0.532 N=1.400 K=0.001.csv" #

        # Generate the file path dynamically
        if EPS>1:
            tmatrix_file_dir = os.path.join(
                folder_path,
                f"LAM={wavelength_str} EPS={EPS_str} k={ri_k_str}",
                f"RANDOMLY ORIENTED OBLATE SPHEROIDS EPS={EPS_str} lambda={wavelength_str} N={ri_n_str} K={ri_k_str}.csv"
            )
        else:
            tmatrix_file_dir = os.path.join(
                folder_path,
                f"LAM={wavelength_str} EPS={EPS_str} k={ri_k_str}",
                f"RANDOMLY ORIENTED PROLATE SPHEROIDS EPS={EPS_str} lambda={wavelength_str} N={ri_n_str} K={ri_k_str}.csv"
            )


        tmatrix_data = np.genfromtxt(tmatrix_file_dir, skip_header=1, delimiter=',')
        tmatrix_radius = tmatrix_data[:,0]*1000 #nm
        tmatrix_diameter = tmatrix_radius*2 #nm
        cext_data = tmatrix_data[:,1]
        cbksca_data = tmatrix_data[:,3]
        co_cbksca_data = tmatrix_data[:,4]
        cross_cbksca_data = tmatrix_data[:,5]

        return tmatrix_diameter, cext_data, cbksca_data, co_cbksca_data, cross_cbksca_data

def read_sondewnpn_data(file_path, quick_info = False, vertical_offset = 0): # used in tracer_avp
    housondewnpn_file_path = file_path
    housondewnpn_dataset = nc.Dataset(housondewnpn_file_path, 'r')

    # Read variables into their respective numpy arrays
    housondewnpn_rh       = housondewnpn_dataset.variables['rh'][:]
    housondewnpn_temp     = housondewnpn_dataset.variables['tdry'][:] 
    housondewnpn_dp       = housondewnpn_dataset.variables['dp'][:]
    housondewnpn_pressure = housondewnpn_dataset.variables['pres'][:] * 100 # in pa
    housondewnpn_lat      = housondewnpn_dataset.variables['lat'][:]
    housondewnpn_lon      = housondewnpn_dataset.variables['lon'][:]
    housondewnpn_alt      = housondewnpn_dataset.variables['alt'][:] / 1000 # in km

    housondewnpn_base_time    = np.datetime64(int(housondewnpn_dataset.variables['base_time'][:]), 's') 
    housondewnpn_time_offset  = housondewnpn_dataset.variables['time_offset'][:]

    datetimes = housondewnpn_base_time + np.array(housondewnpn_time_offset, dtype='timedelta64[s]')

    housondewnpn_temp = housondewnpn_temp + 273.15 # in K

    if quick_info:
        # ANSI escape code for blue text
        BLUE = '\033[94m'
        # ANSI escape code to reset to default color
        ENDC = '\033[0m'

        # Print dimensions with formatted spacing and color
        print("Dimensions:")
        for name, dimension in housondewnpn_dataset.dimensions.items():
            print(f"  {BLUE}{name:<15}{ENDC} Size: {len(dimension)}")

        # Print variables, their shape, description on a new line, and type
        print("\nVariables:")
        for name, variable in housondewnpn_dataset.variables.items():
            description = getattr(variable, 'long_name', getattr(variable, 'description', 'No description'))
            print(f"  {BLUE}{name:<15}{ENDC} Shape: {variable.shape}, Type: {variable.dtype}")
            print(f"    Description: {description}")
        
        # Create a 1x3 panel plot
        fig, axes = plt.subplots(1, 3, figsize=(9, 6))

        # Pressure vs. Height
        axes[0].plot(housondewnpn_pressure, housondewnpn_alt, color='blue')
        axes[0].set_title('Pressure')
        axes[0].set_xlabel('Pressure (hPa)')
        axes[0].set_ylabel('Altitude (km)')

        # Temperature & Dew Point vs. Height
        axes[1].plot(housondewnpn_temp, housondewnpn_alt, label='Tdry', color='red')
        axes[1].plot(housondewnpn_dp, housondewnpn_alt, label='Dew Point', color='green')
        axes[1].set_title('Temp & DP')
        axes[1].set_xlabel('Temperature (°C)')
        axes[1].legend()

        # Relative Humidity vs. Height
        axes[2].plot(housondewnpn_rh, housondewnpn_alt, color='cyan')
        axes[2].set_title('RH')
        axes[2].set_xlabel('Relative Humidity (%)')

        # Adjust the layout
        plt.tight_layout()
        plt.show()

        # Create a new figure with a map projection
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(5, 5))  # Ensuring a square figure

        # Set a more focused extent around Houston [lon_min, lon_max, lat_min, lat_max]
        ax.set_extent([-96.5, -94.5, 28.5, 30.5])

        # Major roads
        roads = cfeature.NaturalEarthFeature(
            category='cultural',
            name='roads',
            scale='10m',
            facecolor='none')
        road_feature = ax.add_feature(roads, edgecolor='darkgray')

        # Add natural features for better visualization
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle='-')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.LAKES, edgecolor='black', facecolor=cfeature.COLORS['water'])
        ax.add_feature(cfeature.RIVERS)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')  # Making the ocean blue

        # Plot the latitude and longitude, colored by altitude using 'viridis' colormap
        sc = ax.scatter(housondewnpn_lon, housondewnpn_lat, c=housondewnpn_alt, cmap='cividis_r', s=1, transform=ccrs.PlateCarree())
        sc.set_zorder(3)  # Ensuring scatter is above all other features

        ax.scatter(-95.0424, 29.3884, zorder=4)

        # Ensure roads are above natural features but below scatter points
        road_feature.set_zorder(1)


        # Add colorbar inside the map with more ticks
        axins = ax.inset_axes([0.76, 0.03, 0.03, 0.3])  # [x, y, width, height]
        ticks = np.linspace(housondewnpn_alt.min(), housondewnpn_alt.max(), 5)  # Generate 5 evenly spaced ticks
        cbar = fig.colorbar(sc, cax=axins, orientation='vertical', ticks=ticks)
        cbar.set_label("km", rotation=0, labelpad=10)

        # Display the map
        plt.show()
    
    # Close the dataset when done
    housondewnpn_dataset.close()

    pressure_interpolator    = scipy.interpolate.interp1d(housondewnpn_alt+vertical_offset, housondewnpn_pressure, bounds_error=False, fill_value=np.nan)
    temperature_interpolator = scipy.interpolate.interp1d(housondewnpn_alt+vertical_offset, housondewnpn_temp, bounds_error=False, fill_value=np.nan)
    rh_interpolator          = scipy.interpolate.interp1d(housondewnpn_alt+vertical_offset, housondewnpn_rh, bounds_error=False, fill_value=np.nan)

    return pressure_interpolator, temperature_interpolator, rh_interpolator, datetimes

def read_arm_tropoe_sounding_data(file_path, start_time=None, end_time=None):
    # Load data
    dataset = nc.Dataset(file_path, 'r')
    
    # Read base time and time offset
    base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
    time_offsets = dataset.variables['time_offset'][:]
    times = base_time + np.array(time_offsets, dtype='timedelta64[s]')
    
    # Convert start_time and end_time to np.datetime64 if specified
    if start_time and end_time:
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        time_indices = np.where((times >= start_time) & (times <= end_time))[0]
        if len(time_indices) == 0:
            print("No data found in the specified time range.")
            dataset.close()
            return
    else:
        time_indices = np.arange(len(times))
    
    # Variables with time and height dimensions
    height = dataset.variables['height'][:]
    temperature = dataset.variables['temperature'][time_indices, :]
    pressure = dataset.variables['pressure'][time_indices, :]
    water_vapor = dataset.variables['waterVapor'][time_indices, :]
    relative_humidity = dataset.variables['rh'][time_indices, :]
    
    # Compute time-averaged profiles
    temperature_avg = np.mean(temperature, axis=0)
    pressure_avg = np.mean(pressure, axis=0)
    water_vapor_avg = np.mean(water_vapor, axis=0)
    rh_avg = np.mean(relative_humidity, axis=0)

    # Plotting the averaged profiles
    fig, ax = plt.subplots(1, 4, figsize=(16, 6))

    ax[0].plot(pressure_avg, height)
    ax[0].set_title('Average Pressure')
    ax[0].set_xlabel('Pressure (hPa)')
    ax[0].set_ylabel('Height (km)')

    ax[1].plot(temperature_avg, height, label='Temperature')
    ax[1].set_title('Average Temperature')
    ax[1].set_xlabel('Temperature (°C)')

    ax[2].plot(water_vapor_avg, height)
    ax[2].set_title('Average Water Vapor Mixing Ratio')
    ax[2].set_xlabel('Mixing Ratio (g/kg)')

    ax[3].plot(rh_avg, height)
    ax[3].set_title('Average Relative Humidity')
    ax[3].set_xlabel('RH (%)')

    plt.tight_layout()
    plt.show()
    
    # Close the dataset
    dataset.close()
    
    # Return averaged profiles
    return temperature_avg, pressure_avg, water_vapor_avg, rh_avg

def read_houcsphotaod_data(houcsphotaod_file_paths, start_datetime, end_datetime):
    # Check if the input is a single string, if so, convert it to a list
    if isinstance(houcsphotaod_file_paths, str):
        houcsphotaod_file_paths = [houcsphotaod_file_paths]

    # Initializing accumulators for concatenated data
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_aod_data = {wavelength: [] for wavelength in 
                   ['aod_1640', 'aod_1020', 'aod_870', 'aod_675', 
                    'aod_500', 'aod_440', 'aod_380', 'aod_340', 'angstrom340_440'
                    , 'angstrom380_500', 'angstrom440_870', 'angstrom500_870']}

    # Iterating over each file
    for file_path in houcsphotaod_file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Converting base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offset = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offset, dtype='timedelta64[s]')

            # Concatenating datetimes
            all_datetimes = np.concatenate((all_datetimes, datetimes))

            # Concatenating AOD data
            for wavelength in all_aod_data.keys():
                all_aod_data[wavelength].extend(dataset.variables[wavelength][:])

    # Converting lists to numpy arrays
    for wavelength in all_aod_data.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            all_aod_data[wavelength] = np.array(all_aod_data[wavelength])

    # Filtering based on datetime
    indices = np.where((all_datetimes >= start_datetime) & (all_datetimes <= end_datetime))[0]

    # Creating the final data dictionary
    filtered_houcsphotaod_data = {wavelength: all_aod_data[wavelength][indices] for wavelength in all_aod_data}
    filtered_houcsphotaod_data['datetimes'] = all_datetimes[indices]

    return filtered_houcsphotaod_data

def read_mergedsmpsaps_data(file_paths, start_datetime, end_datetime):
    """
    Reads aerosol data from specified netCDF files within a given datetime range, adjusting
    the start and end times to include the full hours if they fall within those hours.

    Parameters:
    file_paths: list of str or str
        File paths to the netCDF files.
    start_datetime: datetime64 or str
        Start datetime for filtering the data.
    end_datetime: datetime64 or str
        End datetime for filtering the data.

    Returns:
    dict: A dictionary containing filtered data arrays for each variable of interest.
    """
    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Initialize accumulators
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_variables = {}  # Example variable accumulator initialization

    # Define variables of interest
    variables_of_interest = ['time', 'merged_diameter_mobility', 'diameter_aerodynamic', 
                             'diameter_mobility', 'effective_density', 'merged_dN_dlogDp', 
                             'merged_total_N_conc', 'merged_total_SA_conc', 'merged_total_V_conc',
                             'qc_effective_density', 'qc_merged_dN_dlogDp', 'qc_merged_total_N_conc',
                             'qc_merged_total_SA_conc', 'qc_merged_total_V_conc']

    # Initialize data structure for variables
    for var in variables_of_interest:
        all_variables[var] = []

    # Process start and end datetime to ensure full hour coverage
    start_datetime = np.datetime64(start_datetime).astype('datetime64[h]')
    end_datetime = np.datetime64(end_datetime)
    # If end_datetime is not exactly on the hour, round up to the next hour
    if end_datetime != end_datetime.astype('datetime64[h]'):
        end_datetime = end_datetime + np.timedelta64(1, 'h')
        end_datetime = end_datetime.astype('datetime64[h]')

    # Process each file
    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Convert base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offsets = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offsets, dtype='timedelta64[s]')

            # Filter by datetime range
            valid_indices = np.where((datetimes >= start_datetime) & (datetimes < end_datetime))[0]

            # If no valid datetimes, skip to the next file
            if len(valid_indices) == 0:
                continue

            # Store datetimes
            all_datetimes = np.concatenate([all_datetimes, datetimes[valid_indices]])

            # Loop through variables of interest and store their data
            for var in variables_of_interest:
                if var in dataset.variables:
                    data = dataset.variables[var][:]
                    # For variables with a time dimension, filter by valid_indices
                    if 'time' in dataset.variables[var].dimensions:
                        filtered_data = data[valid_indices]
                    else:
                        filtered_data = data  # For static variables, no filtering needed
                    all_variables[var].extend(filtered_data.tolist())

    # Convert lists to numpy arrays for easier handling
    for var in all_variables:
        all_variables[var] = np.array(all_variables[var])

    # Include datetimes in the results
    all_variables['datetimes'] = all_datetimes

    return all_variables

def read_arm_ccn_data(file_paths, start_datetime, end_datetime):
    """
    Reads CCN and atmospheric data from specified netCDF files within a given datetime range,
    rounding the start time to the previous hour and the end time to the next hour.

    Parameters:
    file_paths : list of str or str
        File paths to the netCDF files.
    start_datetime : datetime64 or str
        Start datetime for filtering the data.
    end_datetime : datetime64 or str
        End datetime for filtering the data.

    Returns:
    dict : A dictionary containing filtered data arrays for each variable of interest.
    """
    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Initialize accumulators
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_variables = {}

    # Define variables of interest
    variables_of_interest = [
        'time', 'droplet_size', 'eta_lookup_table', 'seconds_after_transition', 
        'supersaturation_set_point', 'dT_target_estimated', 'eta_target', 
        'reported_temperature_gradient', 'supersaturation_calculated_target', 
        'eta', 'T_read_gradient', 'supersaturation_calculated', 'temp_unstable', 
        'temperature_std', 'T_set_gradient', 'T_set_TEC1', 'T_read_TEC1', 
        'T_set_TEC2', 'T_read_TEC2', 'T_set_TEC3', 'T_read_TEC3', 'T_nafion', 
        'T_inlet', 'T_OPC', 'dT_OPC', 'T_sample', 'Q_sample', 'Q_sheath', 
        'P_sample', 'laser_current', 'overflow', 'first_stage_monitor_voltage', 
        'N_CCN_bin_number', 'proportional_valve_voltage', 'first_bin_used', 'N_CCN', 
        'qc_N_CCN', 'N_CCN_dN'
    ]

    # Convert start and end datetime to ensure full hour coverage
    start_datetime = np.datetime64(start_datetime)
    end_datetime = np.datetime64(end_datetime)

    # Round down start_datetime to the previous hour and end_datetime to the next hour
    start_datetime = start_datetime.astype('datetime64[h]')
    end_datetime = (end_datetime + np.timedelta64(1, 'h')).astype('datetime64[h]')

    # Process each file
    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Convert base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offsets = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offsets, dtype='timedelta64[s]')

            # Filter by datetime range
            valid_indices = np.where((datetimes >= start_datetime) & (datetimes < end_datetime))[0]

            # Store datetimes
            all_datetimes = np.concatenate([all_datetimes, datetimes[valid_indices]])

            # Loop through variables of interest and store their data
            for var in variables_of_interest:
                if var in dataset.variables:
                    data = dataset.variables[var][:]
                    if 'time' in dataset.variables[var].dimensions:
                        filtered_data = data[valid_indices]
                    else:
                        filtered_data = data
                    all_variables.setdefault(var, []).extend(filtered_data.tolist())

    # Convert lists to numpy arrays
    for var in all_variables:
        all_variables[var] = np.array(all_variables[var])

    # Include datetimes in the results
    all_variables['datetimes'] = all_datetimes

    return all_variables

def read_ccn_spectra_data(file_paths, start_datetime, end_datetime):
    """
    Reads CCN and atmospheric data from specified netCDF files within a given datetime range,
    rounding the start time to the previous hour and the end time to the next hour.
    Takes 'supersaturation_setpoint' from the first file only.

    Parameters:
    file_paths : list of str
        File paths to the netCDF files.
    start_datetime : datetime64 or str
        Start datetime for filtering the data.
    end_datetime : datetime64 or str
        End datetime for filtering the data.

    Returns:
    dict : A dictionary containing concatenated and filtered data arrays for each variable of interest,
           except for 'supersaturation_setpoint', which is taken from the first file only.
    """
    
    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Initialize accumulators
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_variables = {}

    # Variables of interest
    variables_of_interest = [
        'time', 'time_bounds', 'setpoint_time', 
        'supersaturation_calculated', 'N_CCN', 'qc_N_CCN', 'N_CCN_fit_coefs', 
        'N_CCN_fit_error', 'N_CCN_fit_value', 'concentration', 'f_CCN', 
        'qc_f_CCN', 'lat', 'lon', 'alt'
        # Note: 'supersaturation_setpoint' is intentionally omitted here
    ]

    # Convert start and end datetime to ensure full hour coverage
    start_datetime = np.datetime64(start_datetime)
    end_datetime = np.datetime64(end_datetime)
    start_datetime = start_datetime.astype('datetime64[h]')
    end_datetime = (end_datetime + np.timedelta64(1, 'h')).astype('datetime64[h]')

    # Flag to check if 'supersaturation_setpoint' has been read
    supersaturation_setpoint_read = False

    # Process each file
    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Convert base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offsets = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offsets, dtype='timedelta64[s]')

            # Filter by datetime range
            valid_indices = np.where((datetimes >= start_datetime) & (datetimes < end_datetime))[0]
            
            # Store datetimes
            all_datetimes = np.concatenate([all_datetimes, datetimes[valid_indices]])

            # Loop through variables of interest and store their data
            for var in variables_of_interest:
                if var in dataset.variables:
                    data = dataset.variables[var][:]
                    if 'time' in dataset.variables[var].dimensions:
                        if data.ndim == 1:
                            filtered_data = data[valid_indices]
                        else:
                            filtered_data = data[valid_indices, :]
                    else:
                        filtered_data = data
                    all_variables.setdefault(var, []).extend(filtered_data.tolist() if data.ndim > 0 else [filtered_data])

            # Read 'supersaturation_setpoint' only from the first file
            if not supersaturation_setpoint_read and 'supersaturation_setpoint' in dataset.variables:
                all_variables['supersaturation_setpoint'] = dataset.variables['supersaturation_setpoint'][:]
                supersaturation_setpoint_read = True

    # Convert lists to numpy arrays for variables that are not static
    for var in all_variables:
        if var not in ['lat', 'lon', 'alt', 'supersaturation_setpoint']:  # These are either static or single-read
            all_variables[var] = np.array(all_variables[var])

    # Include datetimes in the results
    all_variables['datetimes'] = all_datetimes
    all_variables['N_CCN'] = np.array(all_variables['N_CCN'], dtype=float)
    all_variables['supersaturation_calculated'] = np.array(all_variables['supersaturation_calculated'], dtype=float)
    all_variables['concentration'] = np.array(all_variables['concentration'], dtype=float)


    return all_variables

def read_houaossmps_data(houaossmps_file_paths, start_datetime, end_datetime):
    # Check if the input is a single string, convert it to a list if so
    if isinstance(houaossmps_file_paths, str):
        houaossmps_file_paths = [houaossmps_file_paths]

    # Initializing accumulators for concatenated data
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_dN_dlogDp = []
    all_total_N_conc = []
    all_total_SA_conc = []
    all_qc_dN_dlogDp = []
    all_diameter_mobility = []
    all_diameter_mobility_bounds = []

    # Iterating over each file
    for file_path in houaossmps_file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Converting base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offset = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offset, dtype='timedelta64[s]')

            # Concatenating datetimes and other variables
            all_datetimes = np.concatenate((all_datetimes, datetimes))
            all_dN_dlogDp.append(np.array(dataset.variables['dN_dlogDp'][:]))
            all_total_N_conc.append(np.array(dataset.variables['total_N_conc'][:]))
            all_total_SA_conc.append(np.array(dataset.variables['total_SA_conc'][:]))
            all_qc_dN_dlogDp.append(np.array(dataset.variables['qc_dN_dlogDp'][:]))
            all_diameter_mobility = np.array(dataset.variables['diameter_mobility'][:])
            all_diameter_mobility_bounds = np.array(dataset.variables['diameter_mobility_bounds'][:])
            # ... Concatenate other relevant variables

    # Convert lists of arrays to single concatenated arrays
    all_dN_dlogDp = np.concatenate(all_dN_dlogDp, axis=0)
    all_total_N_conc = np.concatenate(all_total_N_conc, axis=0)
    all_total_SA_conc = np.concatenate(all_total_SA_conc, axis=0)
    all_qc_dN_dlogDp = np.concatenate(all_qc_dN_dlogDp, axis=0)

    all_dN_dlogDp[all_dN_dlogDp == -9999.] = np.nan
    # ... Do the same for other variables

    # Filtering based on datetime
    indices = np.where((all_datetimes >= start_datetime) & (all_datetimes <= end_datetime))[0]

    # Creating the final data dictionary
    filtered_houaossmps_data = {
        'datetimes': all_datetimes[indices],
        'dN_dlogDp': all_dN_dlogDp[indices, :],
        'total_N_conc': all_total_N_conc[indices],
        'total_SA_conc': all_total_SA_conc[indices],
        'qc_dN_dlogDp': all_qc_dN_dlogDp[indices, :],
        'diameter_mobility': all_diameter_mobility, # not affected by the date selection
        'diameter_mobility_bounds': all_diameter_mobility_bounds
        # ... Include other filtered variables
    }

    return filtered_houaossmps_data

def read_houaosopc_data(houaosopc_file_paths, start_datetime, end_datetime):
    # Check if the input is a single string, convert it to a list if so
    if isinstance(houaosopc_file_paths, str):
        houaosopc_file_paths = [houaosopc_file_paths]

    # Initializing accumulators for concatenated data
    all_datetimes = np.array([], dtype='datetime64[s]')
    all_dN_dlogDp = []
    all_particle_count = []
    all_flow_sensor = []
    all_diameter_midpoint = []
    all_diameter_midpoint_bounds = []
    
    # ... Initialize other accumulators for relevant variables

    # Iterating over each file
    for file_path in houaosopc_file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Converting base time and time offset to datetime
            base_time = np.datetime64(int(dataset.variables['base_time'][:]), 's')
            time_offset = dataset.variables['time_offset'][:]
            datetimes = base_time + np.array(time_offset, dtype='timedelta64[s]')

            # Concatenating datetimes and other variables
            all_datetimes = np.concatenate((all_datetimes, datetimes))
            all_dN_dlogDp.append(np.array(dataset.variables['dN_dlogDp'][:]))
            all_particle_count.append(np.array(dataset.variables['particle_count'][:]))
            all_flow_sensor.append(np.array(dataset.variables['flow_sensor'][:]))

            all_diameter_midpoint = np.array(dataset.variables['diameter_midpoint'][:])
            all_diameter_midpoint_bounds = np.array(dataset.variables['diameter_midpoint_bounds'][:])
            # ... Concatenate other relevant variables

    # Convert lists of arrays to single concatenated arrays
    all_dN_dlogDp = np.concatenate(all_dN_dlogDp, axis=0)
    all_particle_count = np.concatenate(all_particle_count, axis=0)
    all_flow_sensor = np.concatenate(all_flow_sensor, axis=0)
    all_diameter_midpoint = all_diameter_midpoint
    all_diameter_midpoint_bounds = all_diameter_midpoint_bounds
    

    all_dN_dlogDp[all_dN_dlogDp == -9999.] = np.nan
    # ... Do the same for other variables

    # Filtering based on datetime
    indices = np.where((all_datetimes >= start_datetime) & (all_datetimes <= end_datetime))[0]

    # Creating the final data dictionary
    filtered_houaosopc_data = {
        'datetimes': all_datetimes[indices],
        'dN_dlogDp': all_dN_dlogDp[indices, :],
        'particle_count': all_particle_count[indices],
        'flow_sensor': all_flow_sensor[indices],
        'diameter_midpoint': all_diameter_midpoint,
        'diameter_midpoint_bounds': all_diameter_midpoint_bounds
        # ... Include other filtered variables
    }

    return filtered_houaosopc_data

def read_houaosnephdry1mM1_data(file_paths, start_time=None, end_time=None):
    """
    Read NetCDF data from one or multiple files, concatenate data, generate datetime64 array, 
    and optionally filter data between start and end time.

    :param file_paths: A single path or a list of paths to NetCDF files
    :param start_time: Optional start time for data selection as a string in format 'YYYY-MM-DD HH:MM:SS'
    :param end_time: Optional end time for data selection as a string in format 'YYYY-MM-DD HH:MM:SS'
    :return: Concatenated and optionally filtered data
    """
    # Ensure file_paths is a list
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    all_data = []
    all_times = []

    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Extract base_time and time_offset
            base_time = dataset.variables['base_time'][:]
            time_offset = np.array(dataset.variables['time_offset'][:])  # Time offset in seconds

            # Ensure base_time is a scalar and convert it to numpy.datetime64
            if isinstance(base_time, np.ndarray):
                base_time = base_time.item()  # Convert numpy array to native Python type
            base_time = np.datetime64(int(base_time), 's')

            # Convert base_time and time_offset to datetime64
            time_array = base_time + time_offset.astype('timedelta64[s]')
            all_times.append(time_array)

            # Store data
            data = {}
            for var in dataset.variables:
                if var != 'time_offset':
                    var_data = dataset.variables[var][:]
                    if var_data.ndim > 0:  # Only store non-scalar data
                        var_data = np.where(var_data <= -9999.0, np.nan, var_data)
                        data[var] = var_data[:]
            all_data.append(data)

    # Concatenate all times
    concatenated_times = np.concatenate(all_times)

    # Initialize a dictionary to store concatenated data
    concatenated_data = {}

    # Concatenate data for each variable
    for var in all_data[0]:
        concatenated_data[var] = np.concatenate([data[var] for data in all_data if var in data])

    # Apply time-based filtering if start_time and end_time are provided
    if start_time is not None and end_time is not None:
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        mask = (concatenated_times >= start_time) & (concatenated_times <= end_time)
        filtered_data = {var: concatenated_data[var][mask] for var in concatenated_data}
        return filtered_data, concatenated_times[mask]
    else:
        return concatenated_data, concatenated_times

def read_houaosnephwet1mM1_data(file_paths, start_time=None, end_time=None):
    """
    Read NetCDF data from one or multiple files, replace -9999.0 with NaN, 
    concatenate data, generate datetime64 array, and optionally filter data 
    between start and end time.

    :param file_paths: A single path or a list of paths to NetCDF files
    :param start_time: Optional start time for data selection as a string in format 'YYYY-MM-DD HH:MM:SS'
    :param end_time: Optional end time for data selection as a string in format 'YYYY-MM-DD HH:MM:SS'
    :return: Concatenated and optionally filtered data
    """
    # Ensure file_paths is a list
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    all_data = []
    all_times = []

    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as dataset:
            # Extract base_time and time_offset
            base_time = dataset.variables['base_time'][:]
            time_offset = np.array(dataset.variables['time_offset'][:])  # Time offset in seconds

            # Ensure base_time is a scalar and convert it to numpy.datetime64
            if isinstance(base_time, np.ndarray):
                base_time = base_time.item()  # Convert numpy array to native Python type
            base_time = np.datetime64(int(base_time), 's')

            # Convert base_time and time_offset to datetime64
            time_array = base_time + time_offset.astype('timedelta64[s]')
            all_times.append(time_array)

            # Store data
            data = {}
            for var in dataset.variables:
                if var not in ['time_offset', 'base_time']:
                    var_data = dataset.variables[var][:]
                    if var_data.ndim > 0:  # Only process non-scalar data
                        # Replace -9999.0 with NaN
                        var_data = np.where(var_data <= -9999.0, np.nan, var_data)
                        data[var] = var_data
            all_data.append(data)

    # Concatenate all times
    concatenated_times = np.concatenate(all_times)

    # Initialize a dictionary to store concatenated data
    concatenated_data = {}

    # Concatenate data for each variable
    for var in all_data[0]:
        concatenated_data[var] = np.concatenate([data[var] for data in all_data if var in data])

    # Apply time-based filtering if start_time and end_time are provided
    if start_time is not None and end_time is not None:
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
        mask = (concatenated_times >= start_time) & (concatenated_times <= end_time)
        filtered_data = {var: concatenated_data[var][mask] for var in concatenated_data}
        return filtered_data, concatenated_times[mask]
    else:
        return concatenated_data, concatenated_times

def filter_doearm_filenames_by_datetime(folder, start_datetime, end_datetime, take_previous_and_after = True):
    """
    Filters filenames in a given folder based on timestamps in their names.
    Includes the file closest to and before the start datetime and the file closest to and after the end datetime.

    :param folder: Path to the folder containing the files.
    :param start_datetime: The start datetime as numpy.datetime64.
    :param end_datetime: The end datetime as numpy.datetime64.
    :return: A list of filenames that fall within the specified datetime range, 
             including the file closest to and before the start datetime and closest to and after the end datetime.
    """
    def extract_datetime_from_filename(filename):
        match = re.search(r'\d{8}\.\d{6}', filename)
        if match:
            datetime_str = match.group()
            formatted_datetime_str = f"{datetime_str[:4]}-{datetime_str[4:6]}-{datetime_str[6:8]}T{datetime_str[9:11]}:{datetime_str[11:13]}:{datetime_str[13:15]}"
            return np.datetime64(formatted_datetime_str)
        return None

    # Get all filenames and sort them
    filenames = sorted(os.listdir(folder))

    # Create a list of tuples (datetime, filename)
    datetimes_and_filenames = [(extract_datetime_from_filename(fn), fn) for fn in filenames if extract_datetime_from_filename(fn) is not None]

    # Filter filenames within the datetime range
    filtered_filenames = [fn for dt, fn in datetimes_and_filenames if start_datetime <= dt <= end_datetime]

    if take_previous_and_after:
        # Find the file closest to and before start_datetime and closest to and after end_datetime
        before_start_files = [fn for dt, fn in datetimes_and_filenames if dt < start_datetime]
        after_end_files = [fn for dt, fn in datetimes_and_filenames if dt > end_datetime]

        if before_start_files:
            closest_before_start = min(before_start_files, key=lambda fn: abs(extract_datetime_from_filename(fn) - start_datetime))
            filtered_filenames.insert(0, closest_before_start)

        if after_end_files:
            closest_after_end = min(after_end_files, key=lambda fn: abs(extract_datetime_from_filename(fn) - end_datetime))
            filtered_filenames.append(closest_after_end)

    return [os.path.join(folder, filename) for filename in filtered_filenames]

def filter_aod_data_by_datetime(filepath, start_datetime, end_datetime): # Deappreciated
    """
    Filter AOD data by a specified datetime range.

    :param filepath: Path to the data file.
    :param start_datetime: Start datetime as numpy.datetime64.
    :param end_datetime: End datetime as numpy.datetime64.
    :return: Tuple of filtered datetime_array and AOD_560nm.
    """
    # Read the data from the file
    data_np = np.genfromtxt(filepath, dtype='str', delimiter=',', skip_header=6, filling_values=np.nan)

    # Process AOD data
    AOD_560nm = data_np[1:, 4].astype(float)
    AOD_560nm[AOD_560nm == -999.] = np.nan

    # Format date strings and create datetime array
    formatted_date_array = ['{}-{}-{}'.format(date.split(':')[2], date.split(':')[1], date.split(':')[0]) for date in data_np[1:, 0]]
    datetime_strings = [date + 'T' + time for date, time in zip(formatted_date_array, data_np[1:, 1])]
    datetime_array = np.array(datetime_strings, dtype='datetime64')

    # Filter data by the specified datetime range
    indices = np.where((datetime_array >= start_datetime) & (datetime_array <= end_datetime))[0]
    return datetime_array[indices], AOD_560nm[indices]

def show_dataset_info(dataset):

    # ANSI escape code for blue text
    BLUE = '\033[94m'
    # ANSI escape code to reset to default color
    ENDC = '\033[0m'

    # Print dimensions with formatted spacing and color
    print("Dimensions:")
    for name, dimension in dataset.dimensions.items():
        print(f"  {BLUE}{name:<15}{ENDC} Size: {len(dimension)}")

    # Print variables, their shape, description on a new line, and type
    print("\nVariables:")
    for name, variable in dataset.variables.items():
        description = getattr(variable, 'long_name', getattr(variable, 'description', 'No description'))
        print(f"  {BLUE}{name:<15}{ENDC} Shape: {variable.shape}, Type: {variable.dtype}")
        print(f"    Description: {description}")

def four_panel_size_distribution_plot(diameters, dndlogdp):

    # from dNdlogdp to dSdlogdp
    dSdlogdp_size_dist = np.pi * np.square(diameters) * dndlogdp

    # from dNdlogdp to dVdlogdp
    dVdlogdp_size_dist = np.pi/6 * np.power(diameters, 3) * dndlogdp

    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    ax1[0,0].plot(diameters, dndlogdp)
    ax1[0,0].set_xscale('log')
    ax1[0,0].set_xlim((5, 10E3))
    ax1[0,0].set_title('Number distribution\nlog(x)')

    ax1[0,1].plot(diameters, dndlogdp)
    ax1[0,1].set_xscale('log')
    ax1[0,1].set_yscale('log')
    ax1[0,1].set_xlim((5, 10E3))
    ax1[0,1].set_title('Number distribution\nlog(x), log(y)')

    ax1[1,0].plot(diameters, dSdlogdp_size_dist)
    ax1[1,0].set_xlim((0, 10E3))
    ax1[1,0].set_title('Area distribution\n')
    ax1[1,0].set_xlabel('nm')

    ax1[1,1].plot(diameters, dVdlogdp_size_dist)
    ax1[1,1].set_xlim((0, 10E3))
    ax1[1,1].set_title('Volume distribution\n')
    ax1[1,1].set_xlabel('nm')

    return fig1, ax1

def replace_outliers(data, num_std_dev=1.5, replacement_method='mean'):
    """
    Replace outliers in the data with the mean or median of the dataset.
    
    :param data: The data array.
    :param num_std_dev: Number of standard deviations for determining outliers.
    :param replacement_method: Method for replacing outliers ('mean' or 'median').
    :return: Data array with outliers replaced.
    """
    data = data.copy()
    mean = np.mean(data)
    std_dev = np.std(data)

    # Define the replacement value
    if replacement_method == 'median':
        replacement_value = np.median(data)
    else:
        replacement_value = mean

    # Identify outliers
    outliers_mask = np.abs(data - mean) > num_std_dev * std_dev

    # Replace outliers
    data[outliers_mask] = replacement_value

    return data

def datetime64_to_epoch_seconds(datetime_array, show_info = False):
    # Assuming the datetime_array is in seconds
    return (datetime_array - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')

def read_mie_data(folder_path, n, k):
    n_str = '%.4f' % n
    k_str = '%.4f' % k
    filename_prefix = f"mie_m={n_str}-{k_str}j.csv"
    for file in os.listdir(folder_path):
        if file.startswith(filename_prefix):
            file_path = os.path.join(folder_path, file)
            
            with open(file_path, 'r') as csvfile:
                data_reader = csv.reader(csvfile)
                headers = next(data_reader)

                # Initialize lists to store column data
                diameter, Qext, Qsca, Qback, g = [], [], [], [], []

                for row in data_reader:
                    diameter.append(float(row[0]))
                    Qext.append(float(row[1]))
                    Qsca.append(float(row[2]))
                    Qback.append(float(row[3]))
                    g.append(float(row[4]))

                # Convert lists to numpy arrays
                diameter = np.array(diameter)
                Qext = np.array(Qext)
                Qsca = np.array(Qsca)
                Qback = np.array(Qback)/(4*np.pi)
                g = np.array(g)
                
                return diameter, Qext, Qsca, Qback, g

    raise FileNotFoundError(f"No file found for n={n} and k={k}")


def print_netcdf_info(file_path):
    """
    Print the dimensions, variables, and attributes of a NetCDF file.

    :param file_path: Path to the NetCDF file
    """
    with nc.Dataset(file_path, 'r') as dataset:
        print("Dimensions:")
        for dim_name, dim in dataset.dimensions.items():
            print(f"  {dim_name}: {dim.size}")

        print("\nVariables:")
        for var_name, var in dataset.variables.items():
            print(f"  {var_name}: {var.dtype}, shape={var.shape}")
            print(f"    Attributes:")
            for attr_name in var.ncattrs():
                print(f"      {attr_name}: {var.getncattr(attr_name)}")

        print("\nGlobal Attributes:")
        for attr_name in dataset.ncattrs():
            print(f"  {attr_name}: {dataset.getncattr(attr_name)}")

def robert1999_model2(RH, sigma, a, b):
    # for testing correction of the aerosol hygroscopic growth
    return sigma * (1 + a * np.power(RH/100, b))

def robert1999_model3(RH, sigma, a, b, c, d, g):
    # for testing correction of the aerosol hygroscopic growth
    first_term  = (1 + a*np.power(RH/100, b)) * (1 - switch_function(RH, d))
    second_term = c * np.power((1-RH/100), -g) * switch_function(RH, d)
    return(sigma*(first_term+second_term))

def switch_function(RH, d):
    return (np.pi/2 + np.arctan(1E24*(RH/100-d/100)))/np.pi

def read_cloud_mask(file_paths, start_time, end_time):
    # Initialize empty arrays for concatenating data
    combined_time = np.array([], dtype='datetime64[s]')
    combined_cloud_mask = np.array([], dtype=np.float32)  # Adjust dtype according to your cloud mask data type
    height = None  # Will be overwritten by the last file's height data

    # Ensure file_paths is a list to simplify processing
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    for file_path in file_paths:
        dataset = nc.Dataset(file_path)

        # Extract variables
        time = dataset.variables['time'][:]
        cloud_mask = dataset.variables['cloud_mask'][:]
        height = dataset.variables['height'][:]

        # Convert time to datetime objects
        base_time = dataset.variables['base_time'][:]
        time_offset = dataset.variables['time_offset'][:]
        datetime_array = np.array([datetime.datetime.utcfromtimestamp(base_time + t) for t in time_offset])

        # Convert datetime_array to numpy.datetime64
        datetime64_array = np.array(datetime_array).astype('datetime64[s]')

        # If processing the first file, initialize the combined arrays
        if combined_time.size == 0:
            combined_time = datetime64_array
            combined_cloud_mask = cloud_mask
        else:
            # Concatenate the new data with the existing arrays
            combined_time = np.concatenate((combined_time, datetime64_array))
            combined_cloud_mask = np.concatenate((combined_cloud_mask, cloud_mask))

    # Filter the combined data based on the start_time and end_time
    mask = (combined_time >= np.datetime64(start_time)) & (combined_time <= np.datetime64(end_time))
    filtered_time = combined_time[mask]
    filtered_cloud_mask = combined_cloud_mask[mask]

    return filtered_time, filtered_cloud_mask, height

def reformat_date(date_str):
    """Convert date from 'MM/DD/YYYY' to 'YYYY-MM-DD' format."""
    return datetime.datetime.strptime(date_str, '%m/%d/%Y').strftime('%Y-%m-%d')

def convert_to_24hr(time_str, am_pm):
    """Convert 12-hour clock format to 24-hour clock format."""
    return datetime.datetime.strptime(f"{time_str} {am_pm}", '%I:%M:%S %p').strftime('%H:%M:%S')

def parse_tamu_sounding_data(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()

    datetimes = []
    pressures = []
    temperatures = []
    rhs = []
    altitudes = []  # List to store altitude data

    data_start = False
    for line in lines:
        if 'FltTime' in line and 'Press' in line:
            data_start = True
            continue

        if data_start and line.strip() and not line.startswith('----'):
            fields = line.split()

            # Check if the line has enough fields and if the first field is a number
            if len(fields) < 29 or not fields[0].replace('.', '', 1).isdigit():
                continue

            try:
                # Convert and combine date and time
                date_str = reformat_date(fields[22])  # Reformat 'UTC_Date' column
                time_str = fields[23]                # 'UTC_Time' column
                am_pm = fields[24]                   # AM/PM indicator
                time_24hr_str = convert_to_24hr(time_str, am_pm)
                datetime_str = f"{date_str} {time_24hr_str}"
                datetime_np = np.datetime64(datetime_str)
            except Exception as e:
                print(f"Error in date-time conversion: {e}")
                continue

            try:
                # Parsing other data fields
                pressure = float(fields[1])
                temperature = float(fields[2].replace('+', ''))
                rh = float(fields[3])
                altitude = float(fields[8])  # Adjust index as needed for altitude column
            except Exception as e:
                print(f"Error in data parsing: {e}")
                continue

            datetimes.append(datetime_np)
            pressures.append(pressure)
            temperatures.append(temperature)
            rhs.append(rh)
            altitudes.append(altitude)  # Append altitude data

    return np.array(datetimes), np.array(pressures), np.array(temperatures), np.array(rhs), np.array(altitudes)

def chose_tamu_sounde_file_name(folder, start_time, end_time):

    """
    Finds the file in the specified folder whose timestamp is closest to either the start or end time.

    Args:
    folder (str): The folder path containing the files.
    start_time (np.datetime64): The start time.
    end_time (np.datetime64): The end time.

    Returns:
    str: The name of the closest file.
    """

    # Convert numpy datetime64 to Python datetime for easier manipulation
    start_time = start_time.astype(datetime.datetime)
    end_time = end_time.astype(datetime.datetime)

    mid_time = start_time + (end_time - start_time) / 2

    closest_file = None
    min_time_diff = np.inf

    # Iterate through files in the folder
    for file in os.listdir(folder):
        if file.startswith('TAMU_TRACER') and file.endswith('.txt'):
            # Extract the timestamp from the file name
            timestamp_str = file.split('_')[2] + ' ' + file.split('_')[3]
            file_time = datetime.datetime.strptime(timestamp_str, '%Y%m%d %H%M')

            # Calculate time difference to the midpoint
            time_diff_to_mid = abs((file_time - mid_time).total_seconds())

            # Check if this file is closer to the midpoint than the current closest
            if time_diff_to_mid < min_time_diff:
                min_time_diff = time_diff_to_mid
                closest_file = file

    return os.path.join(folder, closest_file)

def calculate_angstrom(tau1, tau2, lambda1, lambda2):
    angstrom = - np.log(tau1/tau2) / np.log(lambda1/lambda2)
    return angstrom

def calculate_tau2(tau1, lambda1, lambda2, angstrom):
    tau2 = np.power((lambda2/lambda1), -angstrom) * tau1
    return tau2


def extract_netcdf_data(file_path):
    """
    Extracts all variable data from a NetCDF file and returns them as a dictionary,
    including calculated datetime values as np.datetime64.

    Parameters:
    - file_path: str, path to the NetCDF file

    Returns:
    - data_dict: dict, dictionary containing all variable data including 'datetime' as np.datetime64
    """
    data_dict = {}

    # Open the NetCDF file
    with nc.Dataset(file_path, 'r') as ncfile:
        # Extract variables and their data
        for var_name in ncfile.variables:
            var = ncfile.variables[var_name]
            var_data = var[:]
            data_dict[var_name] = var_data

        # Calculate actual datetime values if 'base_time' and 'time_offset' exist
        if 'base_time' in data_dict and 'time_offset' in data_dict:
            base_time = data_dict['base_time']
            time_offset = data_dict['time_offset']
            
            # Convert base_time to a datetime object
            base_datetime = np.datetime64('1970-01-01T00:00:00') + np.timedelta64(int(base_time), 's')
            
            # Calculate actual datetime values
            actual_time = base_datetime + time_offset.astype('timedelta64[s]')
            data_dict['datetime'] = actual_time

    return data_dict


def get_time_x(target_datetime, interpolated_datetime):
    # Convert datetimes to seconds
    interp_seconds = datetime_to_seconds(interpolated_datetime)
    target_seconds = (target_datetime - interpolated_datetime[0]).astype('timedelta64[s]').astype(int)
    # Interpolate to find the corresponding index
    interpolated_index = np.interp(target_seconds, interp_seconds, np.arange(len(interpolated_datetime)))
    return interpolated_index


def datetime_to_seconds(dt_array):
    return (dt_array - dt_array[0]).astype('timedelta64[s]').astype(int)


def read_sondewnpn_data_for_BRN(file_path, quick_info=False):

    housondewnpn_file_path = file_path
    housondewnpn_dataset = nc.Dataset(housondewnpn_file_path, 'r')

    # Read variables into their respective numpy arrays
    housondewnpn_rh = housondewnpn_dataset.variables['rh'][:]
    housondewnpn_temp = housondewnpn_dataset.variables['tdry'][:] + 273.15  # Convert to Kelvin
    housondewnpn_pressure = housondewnpn_dataset.variables['pres'][:] * 100  # Convert to Pa
    housondewnpn_u_wind = housondewnpn_dataset.variables['u_wind'][:]
    housondewnpn_v_wind = housondewnpn_dataset.variables['v_wind'][:]
    housondewnpn_alt = housondewnpn_dataset.variables['alt'][:]

    # Read time information
    housondewnpn_base_time = np.datetime64(int(housondewnpn_dataset.variables['base_time'][:]), 's')
    housondewnpn_time_offset = housondewnpn_dataset.variables['time_offset'][:]
    datetimes = housondewnpn_base_time + np.array(housondewnpn_time_offset, dtype='timedelta64[s]')

    if quick_info:
        # ANSI escape code for blue text
        BLUE = '\033[94m'
        ENDC = '\033[0m'

        # Print dimensions with formatted spacing and color
        print("Dimensions:")
        for name, dimension in housondewnpn_dataset.dimensions.items():
            print(f"  {BLUE}{name:<15}{ENDC} Size: {len(dimension)}")

        # Print variables, their shape, description on a new line, and type
        print("\nVariables:")
        for name, variable in housondewnpn_dataset.variables.items():
            description = getattr(variable, 'long_name', getattr(variable, 'description', 'No description'))
            print(f"  {BLUE}{name:<15}{ENDC} Shape: {variable.shape}, Type: {variable.dtype}")
            print(f"    Description: {description}")

    # Close the dataset when done
    housondewnpn_dataset.close()

    # Return the necessary variables for bulk Richardson number calculation
    return housondewnpn_alt, housondewnpn_temp, housondewnpn_pressure, housondewnpn_u_wind, housondewnpn_v_wind, datetimes