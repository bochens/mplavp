# Standard library imports
import os
import warnings
from itertools import cycle
from datetime import datetime, timezone

# Scientific and numerical computing
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, zscore
from scipy.optimize import curve_fit

# Plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter, PercentFormatter

# NetCDF handling
import netCDF4 as nc

# Local project modules
import find_layers
import inversempl
import kappa_kohler_theory
import plotmpl
import retrieval_aux
from pympl import PyMPL

# Suppress runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)



TRACER_data_folder = '/Users/bochen/TAMU/tamu_tracer/tracer_data'
ARM_data_folder    = '/Volumes/DRIVE 6/TRACER_DOE_ARM_data'

# Input folder for ARM Retrieval
ARM_mpl_folder           = os.path.join(ARM_data_folder,'houmplpolfs/239067')
ARM_houcsphotaod_folder  = os.path.join(ARM_data_folder,'houcsphotaod/243430')
ARM_radiosounde_folder   = os.path.join(ARM_data_folder,'HOUSSONDEWNPN')
ARM_sizedist_folder      = os.path.join(ARM_data_folder,'houmergedsmpsaps')
ARM_CCN_folder           = os.path.join(ARM_data_folder,'houaosccn2colaspectraM1')
ARM_tropoe_folder        = os.path.join(ARM_data_folder,'houtropoe')
ARM_HTDMA_folder         = os.path.join(ARM_data_folder,'houaoshtdma')

ARM_ACSM_kappa_file      = os.path.join(ARM_data_folder,'acsm_kappa.csv')


# Input folder for TAMU retrieval
tamu_mpl_folder          = os.path.join(TRACER_data_folder,'mpl_files_corrected')
tamu_mpl_ap_file_path    = os.path.join(TRACER_data_folder,'MPL_config_files/MMPL5051_Afterpulse_202302161704_15m_energy_fixed.bin')
tamu_mpl_ov_file_path    = os.path.join(TRACER_data_folder,'MPL_config_files/MMPL5051_Overlap_202302211516_15m_energy_fixed.bin')
tamu_mpl_dt_file_path    = os.path.join(TRACER_data_folder,'MPL_config_files/deadtime_correction_5051_SPCM34394_20230928.csv')
tamu_sizedist_folder     = os.path.join(TRACER_data_folder,'SMPS_POPS_MERGED/SMPS_POPS_MERGED_k=0.001_new')
tamu_radiosounde_folder  = os.path.join(TRACER_data_folder,'TAMU_TRACER_radiosonde_data_final/TSPOTINT')
tamu_CCN_folder          = os.path.join(TRACER_data_folder,'CCN')

# INP concentration
ARM_INP_concentration_folder    = os.path.join(TRACER_data_folder,'INP_Concentrations/inp_conc_amf1')
TAMU_INP_concentration_folder   = os.path.join(TRACER_data_folder,'INP_Concentrations/inp_conc_roamv-')


def process_mpl_data(start_time, end_time, input_folder, 
                     ap_file, ov_file, dt_file,
                     suffix = '*.mpl',
                     time_resolution = 60 ,mov_avg_win = None,
                     scale_num = 30, length_treshold = 25, value_treshold = 1.5, 
                     edge_treshold = 1.4, contain_cloud_treshold = 1.45, cloud_pixel_number_treshold = 2, find_cloud_range = 8,
                     not_a_cloud_treshold = 1.4,
                     snr_treshold = 5,
                     blind_zone = 0.1,
                     fig = None, axs = None, figsize = (15,3.5), plot_range_max = 10,
                     savefig = False, showplot = False, 
                     output_folder = '', fig_name = None):
    
    file_paths  = PyMPL.get_file_list_by_start_end_datetime(input_folder, start_time, end_time)
    print(file_paths)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=np.RankWarning)
        mpl_object = PyMPL(file_paths, ap_file, ov_file, dt_file, blind_range = blind_zone)

    mpl_object.interpolate_data(time_resolution, start_time = np.datetime64(start_time), end_time = np.datetime64(end_time), mov_avg_win=mov_avg_win)
    #first_bin_normalizing = np.nanmean(mpl_object.interpolated_nrb_copol[:, 0])
    first_bin_normalizing = np.mean(mpl_object.interpolated_nrb_copol[~np.isnan(mpl_object.interpolated_nrb_copol[:, 0]) & (np.abs(zscore(mpl_object.interpolated_nrb_copol[:, 0], nan_policy='omit')) < 2), 0])
    normalized_copol_nrb = mpl_object.interpolated_nrb_copol/first_bin_normalizing

    cloud_finding_range_indicies = np.where(mpl_object.range<find_cloud_range)[0]

    cloud_mask, all_cloud_bottoms, all_cloud_tops = find_layers.cwt_cloud_mask(normalized_copol_nrb[:, cloud_finding_range_indicies], mpl_object.range[cloud_finding_range_indicies], \
            scale_num = scale_num, length_treshold = length_treshold, value_treshold = value_treshold, edge_treshold = edge_treshold, \
            contain_cloud_treshold = contain_cloud_treshold, cloud_pixel_number_treshold = cloud_pixel_number_treshold, not_a_cloud_treshold = not_a_cloud_treshold)
    
    # quick fix for high clouds
    high_clouds_finding_start_index = np.where(mpl_object.range>6)[0][0]
    normalized_copol_nrb[:, cloud_finding_range_indicies][:, high_clouds_finding_start_index:] = np.where(normalized_copol_nrb[:, cloud_finding_range_indicies][:, high_clouds_finding_start_index:] > 0.5, np.nan, normalized_copol_nrb[:, cloud_finding_range_indicies][:, high_clouds_finding_start_index:])
    
    cloud_free_column = np.all(cloud_mask != 1, axis=1)

    high_snr_depol = mpl_object.select_snr(mpl_object.interpolated_depol_ratio, mpl_object.interpolated_snr_copol, snr_treshold)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range, normalized_copol_nrb, fig=fig, ax=axs[0], range_max = plot_range_max, vmin=0, vmax=2, x_tick_number = 4, colorbar_label='NRB')
    plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range, high_snr_depol, fig=fig, ax=axs[1], range_max = plot_range_max, vmin=0.01, vmax=1, color_map = plotmpl.lidar_jet, colorbar_norm = 'log', x_tick_number = 4)
    plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range[cloud_finding_range_indicies], cloud_mask, fig=fig, ax=axs[2], range_max = plot_range_max, x_tick_number = 4, colorbar_bool = True)
    

    axs[0].set_xlabel('Time (UTC)', labelpad=-5)
    axs[1].set_xlabel('Time (UTC)', labelpad=-5)
    axs[2].set_xlabel('Time (UTC)', labelpad=-5)

    axs[0].set_ylabel('Altitude AGL (km)')
    axs[1].set_ylabel('Altitude AGL (km)')
    axs[2].set_ylabel('Altitude AGL (km)')

    # for ax, label in zip(axs, ['(a)', '(b)', '(c)']):
    #     ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='w')
    plt.tight_layout()

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_mpl_curtain.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information:
    print('Processing TAMU MPL Data')
    print(f'MPL files: {file_paths}')

    return mpl_object, cloud_mask, cloud_free_column, high_snr_depol, fig, axs


def process_arm_radiosonde_data(start_time, end_time, mpl_range, 
                                input_folder = ARM_radiosounde_folder,
                                vertical_offset = 0,
                                wavelength = 532,
                                fig = None, axs = None, figsize = (9,4),
                                savefig = False, showplot = False, 
                                output_folder = '', fig_name = None):
    '''
    start_time: numpy datetime64
    end_time: numpy datetime64
    '''

    arm_radiosonde_file = retrieval_aux.filter_doearm_filenames_by_datetime(input_folder, start_time, end_time, take_previous_and_after=False)[0]
    pressure_interpolator, temperature_interpolator, rh_interpolator, datetimes = retrieval_aux.read_sondewnpn_data(arm_radiosonde_file, vertical_offset = vertical_offset)
    sonde_beta2  = inversempl.rayleigh_bcksca_coeff(0.532, mpl_range, pressure_interpolater=pressure_interpolator, temperature_interpolater=temperature_interpolator)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    axs[0].plot(rh_interpolator(mpl_range), mpl_range, c='k')
    axs[0].set_xlabel('RH')
    axs[0].set_ylabel('Height (Km)')
    axs[0].set_xlim((0,100))
    axs[0].set_ylim((0,20))
    #axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,2))

    axs[1].plot(pressure_interpolator(mpl_range), mpl_range, c='k')
    axs[1].set_xlabel('Pressure (Pa)')
    axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    axs[1].set_xlim((0,103000))
    axs[1].set_ylim((0,20))

    axs[2].plot(temperature_interpolator(mpl_range), mpl_range, c='k')
    axs[2].set_xlabel('Temperature (K)')
    axs[2].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    axs[2].set_xlim((180,350))
    axs[2].set_ylim((0,20))

    plt.tight_layout()
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_armradiodone_profile.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM Radiosonde Data')
    print(f'ARM Radiosonde Files: {arm_radiosonde_file}')
    print(f'ARM Raidosonde Launch Time is {datetimes[0]}')
    if showplot:
        plt.show()

    return sonde_beta2, rh_interpolator, fig, axs



def process_arm_tropoe_profile(start_time, end_time, mpl_range,
                               input_folder = ARM_tropoe_folder, wavelength=532,
                               fig=None, axs=None, figsize=(9, 4),
                               savefig=False, showplot=False, 
                               output_folder='', fig_name=None):
    '''
    Processes and optionally plots ARM TROPoe profile data within a specified time range and returns interpolators and plot objects.

    Parameters:
    start_time : numpy.datetime64
        The starting time for data retrieval.
    end_time : numpy.datetime64
        The ending time for data retrieval.
    mpl_range : list or array
        The range of MPL data to process.
    input_folder : str
        Path to the folder containing ARM TROPoe files.
    wavelength : int, optional
        Wavelength for data processing, default is 532 nm.
    fig, axs : matplotlib.figure.Figure, matplotlib.axes._axes.Axes, optional
        Figure and axes for plotting, if not provided, a new figure will be created.
    figsize : tuple, optional
        Size of the figure, default is (9, 4).
    savefig : bool, optional
        Whether to save the figure, default is False.
    showplot : bool, optional
        Whether to display the plot, default is False.
    output_folder : str, optional
        Folder to save the figure if savefig is True.
    fig_name : str, optional
        Name of the saved figure file.

    Returns:
    pressure_interpolator, temperature_interpolator, rh_interpolator : scipy.interpolate.interp1d objects
        Interpolators for pressure, temperature, and relative humidity over the height range.
    fig, axs : matplotlib.figure.Figure, matplotlib.axes._axes.Axes
        Figure and axes objects containing the plot.
    '''
    # Identify the ARM radiosonde files within the specified time range
    try:
        arm_radiosonde_files = retrieval_aux.filter_doearm_filenames_by_datetime(
            input_folder, start_time, end_time, take_previous_and_after=True
        )
    except IndexError:
        print("No ARM radiosonde files found in the specified time range.")
        return None

    print(arm_radiosonde_files)

    # Initialize lists to hold data across multiple files
    datetime_list = []
    pressure_list = []
    temperature_list = []
    rh_list = []

    # Loop through each file and concatenate data
    for file in arm_radiosonde_files:
        # Load data using retrieval_aux to extract NetCDF data into a dictionary
        data_dict = retrieval_aux.extract_netcdf_data(os.path.join(input_folder, file))
        
        # Initialize the base datetime and time offset
        base_datetime = np.datetime64(data_dict["base_time"].item(), 's')  # 's' stands for seconds
        time_offset = np.array(data_dict["time_offset"])
        
        # Create datetime array for the current file
        datetime_array = base_datetime + time_offset.astype('timedelta64[s]')
        
        # Append data from the current file to lists
        datetime_list.append(datetime_array)
        pressure_list.append(data_dict['pressure'])
        temperature_list.append(data_dict['temperature'])
        rh_list.append(data_dict['rh'])

    # Concatenate data along the time dimension
    datetime_array = np.concatenate(datetime_list)
    pressure = np.concatenate(pressure_list, axis=0)
    temperature = np.concatenate(temperature_list, axis=0)
    relative_humidity = np.concatenate(rh_list, axis=0)
    height = data_dict['height']  # Assuming height is the same for all files
    print(height)

    # Filter time range based on start_time and end_time
    time_indices = np.where((datetime_array >= start_time) & (datetime_array <= end_time))[0]
    if len(time_indices) == 0:
        print("No data in the specified time range.")
        return None
    
    # Compute time-averaged profiles
    pressure_avg = np.mean(pressure[time_indices, :], axis=0)   *100
    temperature_avg = np.mean(temperature[time_indices, :], axis=0) + 273.15
    rh_avg = np.mean(relative_humidity[time_indices, :], axis=0)
    
    # Create interpolators
    pressure_interpolator = scipy.interpolate.interp1d(height, pressure_avg, bounds_error=False, fill_value=np.nan)
    temperature_interpolator = scipy.interpolate.interp1d(height, temperature_avg, bounds_error=False, fill_value=np.nan)
    rh_interpolator = scipy.interpolate.interp1d(height, rh_avg, bounds_error=False, fill_value=np.nan)

    # Calculate Rayleigh backscatter coefficient
    sonde_beta2 = inversempl.rayleigh_bcksca_coeff(
        wavelength / 1000, mpl_range, pressure_interpolater=pressure_interpolator, temperature_interpolater=temperature_interpolator
    )

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    axs[0].plot(rh_interpolator(mpl_range), mpl_range, c='k')
    axs[0].set_xlabel('RH')
    axs[0].set_ylabel('Height (Km)')
    axs[0].set_xlim((0, 100))
    axs[0].set_ylim((0, 20))

    axs[1].plot(pressure_interpolator(mpl_range), mpl_range, c='k')
    axs[1].set_xlabel('Pressure (Pa)')
    axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0, 2))
    axs[1].set_xlim((0, 103000))
    axs[1].set_ylim((0, 20))

    axs[2].plot(temperature_interpolator(mpl_range), mpl_range, c='k')
    axs[2].set_xlabel('Temperature (K)')
    axs[2].ticklabel_format(style='sci', axis='both', scilimits=(0, 2))
    axs[2].set_xlim((180, 350))
    axs[2].set_ylim((0, 20))

    plt.tight_layout()
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_armradiosonde_profile.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM Radiosonde Data')
    print(f'ARM Radiosonde Files: {arm_radiosonde_files}')
    print(f'ARM Radiosonde Launch Time is {datetime_array[0]}')
    if showplot:
        plt.show()

    return sonde_beta2, rh_interpolator, fig, axs



def process_tamu_radiosonde_data(start_time, end_time, mpl_range,
                                 wavelength = 532,
                                 input_folder = tamu_radiosounde_folder,
                                 fig = None, axs = None, figsize = (15,4),
                                 savefig = False, showplot = False, 
                                 output_folder = '', fig_name = None):
    
    rs_file = retrieval_aux.chose_tamu_sounde_file_name(input_folder, start_time, end_time)
    rs_datetime, rs_pressure, rs_temperature, rs_rh, rs_altitude = retrieval_aux.parse_tamu_sounding_data(rs_file)

    print(rs_file)

    pressure_interpolator    = scipy.interpolate.interp1d(rs_altitude/1000, rs_pressure*100,       bounds_error=False, fill_value=np.nan)
    temperature_interpolator = scipy.interpolate.interp1d(rs_altitude/1000, rs_temperature+273.15, bounds_error=False, fill_value=np.nan)
    rh_interpolator          = scipy.interpolate.interp1d(rs_altitude/1000, rs_rh,                 bounds_error=False, fill_value=np.nan)
    sonde_beta2 = inversempl.rayleigh_bcksca_coeff(wavelength/1000, mpl_range, 
                   pressure_interpolater=pressure_interpolator, temperature_interpolater=temperature_interpolator)


    temperature_profile = temperature_interpolator(mpl_range)-273.15 # in celsius
    saturation_vapor_pressure_profile = 6.112 * np.exp((17.67 * temperature_profile)/(temperature_profile+243.5))
    vapor_pressure = rh_interpolator(mpl_range) * saturation_vapor_pressure_profile
    water_vapor_mixing_ratio = 0.622 * vapor_pressure / (pressure_interpolator(mpl_range) - vapor_pressure)

    potential_temperature = temperature_interpolator(mpl_range) * (pressure_interpolator(mpl_range)/100000)**(287/1004) - 273.15

    
    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=figsize)

    axs[0].plot(rh_interpolator(mpl_range), mpl_range, c='k')
    axs[0].set_xlabel('RH')
    axs[0].set_ylabel('Altitude (Km)')
    axs[0].set_xlim((0,100))
    axs[0].set_ylim((0,20))
    #axs[0].ticklabel_format(style='sci', axis='both', scilimits=(0,2))

    axs[1].plot(pressure_interpolator(mpl_range), mpl_range, c='k')
    axs[1].set_xlabel('Pressure (Pa)')
    axs[1].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    axs[1].set_xlim((0,103000))
    axs[1].set_ylim((0,20))

    axs[2].plot(temperature_interpolator(mpl_range)-273.15, mpl_range, c='k')
    axs[2].set_xlabel('Temperature (C)')
    axs[2].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    #axs[2].set_xlim((180-273.15,350-273.15))
    #axs[2].set_xlim((275-273.15,310-273.15))
    axs[2].set_ylim((0,10))

    axs[3].plot(water_vapor_mixing_ratio, mpl_range, c='k')
    axs[3].set_xlabel('water vapor mixing ratio')
    axs[3].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    axs[3].set_ylim((0,10))

    axs[4].plot(potential_temperature, mpl_range, c='k')
    axs[4].set_xlabel('potential temperature C')
    axs[4].ticklabel_format(style='sci', axis='both', scilimits=(0,2))
    axs[4].set_ylim((0,10))



    plt.tight_layout()
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_tamuradiodone_profile.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing TAMU Radiosonde Data')
    print(f'TAMU Radiosonde Files: {rs_file}')
    print(f'TAMU Raidosonde Launch Time is {rs_datetime[0]}')
    if showplot:
        plt.show()

    return sonde_beta2, rh_interpolator, fig, axs


def process_arm_mergedsizedist_data(start_time, end_time,
                                    input_folder = ARM_sizedist_folder,
                                    figsize=(9,4), fig = None, axs = None,
                                    savefig = False, showplot = False, 
                                    output_folder = '', fig_name = None):
    '''
    return:
        merged_diameter in nm
    '''

    arm_merged_size_file = retrieval_aux.filter_doearm_filenames_by_datetime(input_folder, start_time, end_time, take_previous_and_after=True)

    all_variables            = retrieval_aux.read_mergedsmpsaps_data(arm_merged_size_file, start_time, end_time)
    merged_diameter          = np.array(all_variables['merged_diameter_mobility'], dtype=float) # in nm
    merged_dN_dlogDp         = np.array(all_variables['merged_dN_dlogDp'], dtype=float)
    merged_datetimes         = np.array(all_variables['datetimes'])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_dndlogdp  = np.nanmean(merged_dN_dlogDp, axis=0)
        stdev_dndlogdp = np.nanstd(merged_dN_dlogDp, axis=0)
        n = np.sum(~np.isnan(merged_dN_dlogDp), axis=0)
        sem_dndlogdp = stdev_dndlogdp / np.sqrt(n)

    non_nan_indices = np.where(~np.isnan(mean_dndlogdp))[0]
    total_concentration = scipy.integrate.simpson(y = mean_dndlogdp[non_nan_indices],  x = np.log10(merged_diameter)[non_nan_indices])
    lower_concentration = scipy.integrate.simpson(y = mean_dndlogdp[non_nan_indices]-sem_dndlogdp[non_nan_indices],  x = np.log10(merged_diameter)[non_nan_indices])

    sem_concentration = total_concentration - lower_concentration

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axs[0].set_facecolor('k')
    axs[0].pcolormesh(range(len(merged_datetimes)+1), range(len(merged_diameter)+1), np.transpose(merged_dN_dlogDp))
    _ = axs[0].set_title('ARM merged size distribution')
    axs[0].set_xticks(np.arange(len(merged_datetimes))[::1]+0.5)                # Set X-axis ticks and labels at midpoints
    _ = axs[0].set_xticklabels(merged_datetimes[::1], rotation=0, ha='center')
    axs[0].set_yticks(np.arange(len(merged_diameter))[::10]+0.5)       # Set Y-axis ticks and labels at midpoints
    _ = axs[0].set_yticklabels(merged_diameter[::10], ha='right')
    axs[1].plot(merged_diameter, mean_dndlogdp, color = 'k')
    axs[1].set_xlabel('Diameter (nm)')
    axs[1].set_ylabel('dndlogDp (cm$^{-3}$)')
    axs[1].set_xscale('log')
    axs[1].set_title('Merged Size Distribution')

    plt.tight_layout()
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_armmergedsizedist_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM Size Distribution Data')
    print(f'ARM Sizedistribution Files: {arm_merged_size_file}')
    if showplot:
        plt.show()

    return merged_diameter, mean_dndlogdp, total_concentration, sem_concentration, fig, axs


def process_tamu_mergedsizedist_data(start_time, end_time,
                                     input_folder = tamu_sizedist_folder,
                                     figsize=(15,3), fig = None, axs = None,
                                     number_of_xticks = 5,
                                     savefig = False, showplot = False, 
                                     output_folder = '', fig_name = None):
    
    date_range = PyMPL.get_date_range(start_time, end_time)
    merged_diameter, merged_datetime, merged_ri_n, merged_ri_k, merged_dN_dlogDp \
               = retrieval_aux.read_merged_data(date_range, input_folder)
    merged_select_indices = np.where((merged_datetime >= start_time) & (merged_datetime <= end_time))[0]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_dndlogdp  = np.nanmean(merged_dN_dlogDp[merged_select_indices], axis=0) # did not select the size distribution by correct datetime. Fixed now.
        stdev_dndlogdp = np.nanstd(merged_dN_dlogDp[merged_select_indices], axis=0)
        n = np.sum(~np.isnan(merged_dN_dlogDp[merged_select_indices]), axis=0)
        sem_dndlogdp = stdev_dndlogdp / np.sqrt(n)

    non_nan_indices = np.where(~np.isnan(mean_dndlogdp))[0]
    total_concentration = scipy.integrate.simpson(y = mean_dndlogdp[non_nan_indices],  x = np.log10(merged_diameter)[non_nan_indices])
    lower_concentration = scipy.integrate.simpson(y = mean_dndlogdp[non_nan_indices]-sem_dndlogdp[non_nan_indices],  x = np.log10(merged_diameter)[non_nan_indices])

    sem_concentration = total_concentration - lower_concentration
    

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    axs[0].set_facecolor('k')
    axs[0].pcolormesh(range(len(merged_datetime[merged_select_indices])+1), range(len(merged_diameter)+1), np.transpose(merged_dN_dlogDp[merged_select_indices]))
    _ = axs[0].set_title('TAMU merged size distribution')
    tick_distance = int(len(merged_datetime[merged_select_indices])/number_of_xticks)
    if tick_distance == 0:
        tick_distance = 1
    formatted_times = [datetime.fromtimestamp(d.astype(int), tz=timezone.utc) for d in merged_datetime[merged_select_indices]]
    formatted_times = [d.strftime('%H:%M:%S') for d in formatted_times]
    axs[0].set_xticks(np.arange(len(merged_datetime[merged_select_indices]))[::tick_distance]+0.5)                # Set X-axis ticks and labels at midpoints
    _ = axs[0].set_xticklabels(formatted_times[::tick_distance], rotation=0, ha='center')
    axs[0].set_yticks(np.arange(len(merged_diameter))[::10]+0.5)       # Set Y-axis ticks and labels at midpoints
    _ = axs[0].set_yticklabels(merged_diameter[::10], ha='right')
    axs[1].plot(merged_diameter, mean_dndlogdp, color = 'k')
    axs[1].fill_between(merged_diameter, mean_dndlogdp-sem_dndlogdp, mean_dndlogdp+sem_dndlogdp, color = 'gray', alpha = 0.5)
    axs[1].set_xlabel('Diameter (nm)')
    axs[1].set_ylabel('dndlogDp (cm$^{-3}$)')
    axs[1].set_xscale('log')
    axs[1].set_title('Merged Size Distribution')

    plt.tight_layout()
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_tamumergedsizedist_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)
    
    # Print information
    print('Processing TAMU Size Distribution Data')
    print(f'TAMU Sizedistribution date range: {date_range}')
    if showplot:
        plt.show()

    # === New Figure: Total Number Concentration Time Series (strictly between start_time and end_time, UTC) ===
    concentration_series = []
    times_in_range = []

    for i in range(len(merged_datetime)):
        ts = merged_datetime[i]
        if not (start_time <= ts <= end_time):
            continue

        profile = merged_dN_dlogDp[i]
        valid = ~np.isnan(profile)

        if np.count_nonzero(valid) < 2:
            continue  # skip if not enough valid points

        conc = scipy.integrate.simpson(y=profile[valid], x=np.log10(merged_diameter)[valid])
        
        # Convert np.datetime64 to UTC datetime
        ts_seconds = ts.astype('datetime64[s]').astype(int)
        dt_utc = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)

        concentration_series.append(conc)
        times_in_range.append(dt_utc)

    # Convert start and end time to UTC datetime
    start_seconds = start_time.astype('datetime64[s]').astype(int)
    end_seconds = end_time.astype('datetime64[s]').astype(int)
    start_dt = datetime.fromtimestamp(start_seconds, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_seconds, tz=timezone.utc)

    fig_ts, ax_ts = plt.subplots(figsize=(6, 3))
    ax_ts.plot(times_in_range, concentration_series, color='tab:blue')
    ax_ts.set_title('Total Number Concentration Time Series')
    ax_ts.set_xlabel('Time (UTC)')
    ax_ts.set_ylabel('Total Conc. (cm$^{-3}$)')
    ax_ts.tick_params(axis='x', rotation=45)
    ax_ts.set_xlim(start_dt, end_dt)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.3)

    if savefig:
        ts_fig_name = fig_name.replace('.png', '_timeseries.png') if fig_name else f'{start_time}_{end_time}_total_concentration_timeseries.png'
        fig_ts.savefig(os.path.join(output_folder, ts_fig_name), dpi=300)

    if showplot:
        plt.show()

    return merged_diameter, mean_dndlogdp, sem_dndlogdp, total_concentration, sem_concentration, merged_ri_n, fig, axs


def process_arm_aod_data(start_time, end_time,
                           input_folder = ARM_houcsphotaod_folder,
                           figsize=(6,3), fig = None, axs = None,
                           savefig = False, showplot=False, 
                           output_folder = '', fig_name = None):

    # Angstrom Exponent calculation functions
    def calculate_angstrom(tau1, tau2, lambda1, lambda2):
        angstrom = - np.log(tau1/tau2) / np.log(lambda1/lambda2)
        return angstrom

    def calculate_tau2(tau1, lambda1, lambda2, angstrom):
        tau2 = np.power((lambda2/lambda1), -angstrom) * tau1
        return tau2

    AOD_files = retrieval_aux.filter_doearm_filenames_by_datetime(input_folder, start_time, end_time)
    AOD_data = retrieval_aux.read_houcsphotaod_data(AOD_files, start_time, end_time)
    aod_532 = calculate_tau2(AOD_data['aod_500'], 500, 532, calculate_angstrom(AOD_data['aod_500'], AOD_data['aod_675'], 500, 675))
    mean_aod_532 = np.nanmean(aod_532)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axs.plot(AOD_data['datetimes'], aod_532, color = 'k', marker='o')
    axs.set_xlabel('Datetimes')
    axs.set_ylabel('AOD 532')
    plt.tight_layout()

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_armaod532_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM AOD Data')
    print(f'ARM Sizedistribution Files: {AOD_files}')
    if showplot:
        plt.show()


    return AOD_data['datetimes'], aod_532, mean_aod_532, fig, axs


def process_arm_CCN_data(start_time, end_time,
                                input_folder = ARM_CCN_folder,
                                figsize=(10,4), fig = None, axs = None,
                                savefig = False, showplot = False, 
                                output_folder = '', fig_name = None):
    
    CCN_files = retrieval_aux.filter_doearm_filenames_by_datetime(input_folder, start_time, end_time, take_previous_and_after=True)
    CCN_all_variables     = retrieval_aux.read_ccn_spectra_data(CCN_files, start_time, end_time)
    average_ss_calculated = np.nanmean(CCN_all_variables['supersaturation_calculated'], axis = 0)
    stdev_ss_calculated   = np.nanstd(CCN_all_variables['supersaturation_calculated'], axis = 0)
    average_N_CCN         = np.nanmean(CCN_all_variables['N_CCN'], axis = 0)
    stdev_N_CCN           = np.nanstd(CCN_all_variables['N_CCN'], axis = 0)

    flattened_concentration = np.ravel(CCN_all_variables['concentration'])
    aerosol_concentration = np.nanmean(flattened_concentration, axis = 0)
    stdev_aerosol_concentration = np.nanstd(flattened_concentration, axis=0)
    # Number of observations for standard error calculation
    n_observations_N_CCN = np.sum(~np.isnan(CCN_all_variables['N_CCN']), axis=0)
    stderr_N_CCN = stdev_N_CCN / np.sqrt(n_observations_N_CCN)

    n_observations_concentration = np.sum(~np.isnan(flattened_concentration), axis=0)
    stderr_aerosol_concentration = stdev_aerosol_concentration / np.sqrt(n_observations_concentration)

    print(f'aerosol_concentration.shape {aerosol_concentration.shape}')
    print(aerosol_concentration)
    

    non_nan_indices = np.where(~np.isnan(average_ss_calculated))[0].tolist()
    if non_nan_indices:
        average_ss_calculated = average_ss_calculated[non_nan_indices]
        stdev_ss_calculated   = stdev_ss_calculated[non_nan_indices]
        average_N_CCN         = average_N_CCN[non_nan_indices]
        stdev_N_CCN           = stdev_N_CCN[non_nan_indices]
        stderr_N_CCN          = stderr_N_CCN[non_nan_indices]


    datetimes = CCN_all_variables['datetimes']
    supersaturation_calculated = CCN_all_variables['supersaturation_calculated']
    N_CCN = CCN_all_variables['N_CCN']

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    cmap = plt.cm.viridis  # Colormap for the scatter plot

    # Flatten the data for plotting, handling NaNs efficiently
    
    flat_datetimes = np.repeat(datetimes, [len(ss) for ss in supersaturation_calculated])
    flat_supersaturation = np.concatenate(supersaturation_calculated)
    flat_N_CCN = np.concatenate(N_CCN)

    # Mask to remove any NaNs efficiently from all related arrays
    valid_mask = ~np.isnan(flat_datetimes) & ~np.isnan(flat_supersaturation) & ~np.isnan(flat_N_CCN)
    flat_datetimes = flat_datetimes[valid_mask]
    flat_supersaturation = flat_supersaturation[valid_mask]
    flat_N_CCN = flat_N_CCN[valid_mask]

    # Scatter plot
    sc = axs[0].scatter(flat_datetimes, flat_supersaturation, c=flat_N_CCN, cmap=cmap, norm=mcolors.Normalize(vmin=np.min(flat_N_CCN), vmax=np.max(flat_N_CCN)), s=flat_N_CCN)
    fig.colorbar(sc, ax=axs[0])  # Colorbar for the scatter plot

    # Formatting datetime x-axis
    axs[0].set_xticks(flat_datetimes)
    axs[0].set_xticklabels(flat_datetimes.astype(str))

    # Set labels and titles
    axs[0].set_xlabel('Datetime')
    axs[0].set_ylabel('Supersaturation (%)')
    axs[0].set_title('CCN spectra')

    axs[1].errorbar(average_ss_calculated, average_N_CCN, fmt='.', ls = '-', yerr=stdev_N_CCN, color = 'k', capsize=5, zorder=1)
    axs[1].set_xlabel('Supersaturation (%)')
    axs[1].set_ylabel('CCN Number Concentration (#/cc)')
    axs[1].set_title('average CCN spectra')

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_armCCNspectra_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM CCN spectra Data')
    print(f'ARM CCN spectra Files: {CCN_files}')
    if showplot:
        plt.show()

    return average_ss_calculated, average_N_CCN, stdev_ss_calculated, stdev_N_CCN, stderr_N_CCN, aerosol_concentration, stderr_aerosol_concentration, fig, axs

    
def process_tamu_CCN_data(start_time, end_time,
                          input_folder = tamu_CCN_folder,
                          figsize=(4.5,4), fig=None, axs=None,
                          savefig=False, showplot=False, 
                          output_folder='', fig_name=None):
    
    start_time = pd.to_datetime(start_time).to_pydatetime()
    end_time = pd.to_datetime(end_time).to_pydatetime()

    files = os.listdir(input_folder)
    
    start_date = start_time.date()
    end_date   = end_time.date()
    selected_files = [f for f in files if f.startswith("tracer_tamu_") and f.endswith("_ccn.csv") 
                      and start_date <= datetime.strptime(f.split('_')[2], "%y%m%d").date() <= end_date]
    
    CCN_data = pd.DataFrame()
    for file in selected_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        CCN_data = pd.concat([CCN_data, df], ignore_index=True)

    ccn_datetime = pd.to_datetime(CCN_data['Time'], format='%y%m%d %H:%M:%S')

    time_filtered_CCN_data = CCN_data[(ccn_datetime >= start_time) & (ccn_datetime <= end_time)]

    ss_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

    # Filter data to start from the first row where SS = 0.2 and N_CCN is not NaN
    first_valid_index = time_filtered_CCN_data[(time_filtered_CCN_data['SS'] == 0.2) & 
                                               (~time_filtered_CCN_data['N_CCN'].isna())].index[0]
    filtered_CCN_data = time_filtered_CCN_data.loc[first_valid_index:]

    # Prepare an array to store the mean CCN values for each SS
    average_N_CCN = np.zeros_like(ss_values)
    stdev_N_CCN = np.zeros_like(ss_values)
    sem_N_CCN = np.zeros_like(ss_values)

    # Loop over each SS value, filter data, and calculate the mean
    for i, ss in enumerate(ss_values):
        filtered_data = filtered_CCN_data[filtered_CCN_data['SS'] == ss]
        average_N_CCN[i] = filtered_data['N_CCN'].mean()
        stdev_N_CCN[i] = filtered_data['N_CCN'].std()
        n = len(filtered_data)
        if n > 0:
            sem_N_CCN[i] = stdev_N_CCN[i] / np.sqrt(n)
        else:
            sem_N_CCN[i] = np.nan  # Handle cases where there is no data

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    axs.errorbar(ss_values, average_N_CCN, fmt='.', ls='-', yerr=stdev_N_CCN, color='k', capsize=5, zorder=1)
    
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_tamuCCNspectra_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing TAMU CCN spectra Data')
    print(f'TAMU CCN spectra Files: {selected_files}')
    if showplot:
        plt.show()

    return ss_values, average_N_CCN, stdev_N_CCN, sem_N_CCN, CCN_data, fig, axs
    
    
def process_arm_INP_data(a_time,
                         input_folder = ARM_INP_concentration_folder,
                         figsize=(6,4), fig = None, axs = None,
                         savefig = False, showplot = False, 
                         output_folder = '', fig_name = None):
    
    '''
        axs is a single ax
    '''
    # Use the entire day's INP data
    date_str = np.datetime_as_string(a_time, unit='D').replace('-', '')[2:]
    file_paths = [
        f"{input_folder}_s1.csv",
        f"{input_folder}_s2.csv",
        f"{input_folder}_s3.csv",
        f"{input_folder}_s4.csv"
    ]

    all_temps, all_concentration, inp_temp_list, inp_conc_list, all_relevant_coloumns = INP_process_aux(file_paths, date_str)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axs.plot(all_temps, all_concentration, marker='.', linestyle='-', color='k', label='Summed Concentration')
    axs.set_xlabel('Bin Mids (°C)')
    axs.set_ylabel('Summed INP Concentration (#/L)')

    if savefig:
        if fig_name is None:
            fig_name = f'{date_str}_INPspectra_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing ARM INP spectra Data')
    print(f'ARM INP spectra Files: {all_relevant_coloumns}')
    if showplot:
        plt.show()
    
    return all_temps, all_concentration, inp_temp_list, inp_conc_list, fig, axs


def process_tamu_INP_data(a_time, is_coastal,
                          input_folder = TAMU_INP_concentration_folder,
                          figsize=(6,4), fig = None, axs = None,
                          savefig = False, showplot = False, 
                          output_folder = '', fig_name = None):
    '''
        is_coastal is true or false
    '''

    date_str = np.datetime_as_string(a_time, unit='D').replace('-', '')[2:]
    file_suffix = 'coastal' if is_coastal else 'inland'
    file_paths = [
        f"{input_folder}{file_suffix}_s1.csv",
        f"{input_folder}{file_suffix}_s2.csv",
        f"{input_folder}{file_suffix}_s3.csv",
        f"{input_folder}{file_suffix}_s4.csv"
    ]

    all_temps, all_concentration, inp_temp_list, inp_conc_list, all_relevant_coloumns = INP_process_aux(file_paths, date_str)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    axs.plot(all_temps, all_concentration, marker='.', linestyle='-', color='k', label='Summed Concentration')
    axs.set_xlabel('Bin Mids (°C)')
    axs.set_ylabel('Summed INP Concentration (#/L)')

    if savefig:
        if fig_name is None:
            if is_coastal == True:
                fig_name = f'{date_str}_Coastal_INPspectra_data.png'
            else:
                fig_name = f'{date_str}_Inland_INPspectra_data.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    # Print information
    print('Processing TAMU INP spectra Data')
    print(f'TAMU INP spectra Files: {all_relevant_coloumns}')
    if showplot:
        plt.show()
    
    return all_temps, all_concentration, inp_temp_list, inp_conc_list, fig, axs


def INP_process_aux(file_paths, date_str):
    # Initialize a dataframe for summing the data
    summed_data = pd.DataFrame()

    all_relevant_coloumns = []
    
    for a_file_path in file_paths:
        if os.path.exists(a_file_path):
            df = pd.read_csv(a_file_path)
            df.fillna(0, inplace=True)
            relevant_columns = [col for col in df.columns if col.startswith(date_str)]
            all_relevant_coloumns.extend(relevant_columns)

            if relevant_columns:
                df['file_average'] = df[relevant_columns].mean(axis=1)
                if summed_data.empty:
                    summed_data = df[['bin_mids', 'file_average']].rename(columns={'file_average': 'summed_concentration'})
                else:
                    summed_data['summed_concentration'] += df['file_average']

    all_temps = np.array(summed_data['bin_mids'])
    all_concentration = np.array(summed_data['summed_concentration'])

    # Assuming the same temperature bins as in your example
    inp_temp_list = np.array([-10, -15, -20, -25])
    tolerance = 1e-5
    inp_conc_list = np.zeros(len(inp_temp_list))
    for i, temp in enumerate(inp_temp_list):
        # Find the index for the bin mid that matches the temperature
        temp_as_bin_mid = temp + 0.1  # Adjust according to your bin mid calculation logic
        index = np.where(np.isclose(all_temps, temp_as_bin_mid, atol=tolerance))
        if index[0].size > 0:
            inp_conc_list[i] = all_concentration[index[0][0]]

    return all_temps, all_concentration, inp_temp_list, inp_conc_list, all_relevant_coloumns


def process_total_INP_data(file_path, target_datetime, temperature_list):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert the target date to the desired format
    given_date = np.datetime64(target_datetime)
    given_date_only = given_date.astype('datetime64[D]')
    given_date_str = str(given_date_only)
    
    # Ensure all date columns are strings for comparison
    data.columns = data.columns.astype(str)
    
    # Find the matching column
    if given_date_str in data.columns:
        matching_column = given_date_str
    else:
        matching_column = None
    
    # Sort the bin_mids values
    sorted_bin_mids = sorted(data['bin_mids'].unique())
    
    # Find the closest bin_mids values higher than the temperatures
    closest_higher_bin_mids = []
    for temp in temperature_list:
        for bm in sorted_bin_mids:
            if bm > temp:
                closest_higher_bin_mids.append(bm)
                break
    
    # Filter rows where 'bin_mids' matches the closest higher bin_mids
    filtered_data = data[data['bin_mids'].isin(closest_higher_bin_mids)]
    
    # Extract the values from the matching date column
    if matching_column:
        result_list = filtered_data[matching_column].tolist()
    else:
        result_list = []

    return temperature_list, closest_higher_bin_mids, result_list


def plot_aerosol_comprehensive(start_time, end_time,                                # time
                               merged_diameter, mean_dndlogdp,                      # size dist
                               ss_array, average_N_CCN, stdev_N_CCN,                # CCN
                               all_inp_temps, all_inp_conc,                         # INP
                               sem_dndlogdp = None, sem_N_CCN = None,
                               line_color = 'k',
                               figsize=(14,3.5), fig = None, axs = None,
                               savefig = False, showplot = False, 
                               output_folder = '', fig_name = None):
    
    '''
    
        axs needs to be shappe (4,)
    '''
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=figsize)

    formatter = ScalarFormatter(useMathText=True)  # Use LaTeX formatted text
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # Use scientific notation if exponent is outside -1 to 1
    #axs[0].yaxis.set_major_formatter(formatter)
    
    if merged_diameter is not None:
        axs[0].plot(merged_diameter/1000, mean_dndlogdp, c = line_color)
        if sem_dndlogdp is not None:
            axs[0].fill_between(merged_diameter/1000, mean_dndlogdp-sem_dndlogdp*2, mean_dndlogdp+sem_dndlogdp*2, color = 'gray', alpha = 0.7)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Diameter ($\mathrm{\mu}$m)')
        axs[0].set_ylabel('dndlogDp (cm$^{-3}$)')
        axs[0].set_xlim(5E-3, 20)
        axs[0].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))
        

        axs[1].plot(merged_diameter/1000, mean_dndlogdp, c = line_color)
        if sem_dndlogdp is not None:
            axs[1].fill_between(merged_diameter/1000, mean_dndlogdp-sem_dndlogdp*2, mean_dndlogdp+sem_dndlogdp*2, color = 'gray', alpha = 0.7)
        axs[1].set_xscale('log')
        axs[1].set_yscale('log')
        axs[1].set_xlabel('Diameter ($\mathrm{\mu}$m)')
        axs[1].set_ylabel('dndlogDp (cm$^{-3}$)')
        axs[1].set_xlim(5E-3, 20)
        
    if ss_array is not None:
        axs[2].yaxis.set_major_formatter(formatter)
        axs[2].errorbar(ss_array, average_N_CCN, fmt='.', ls = '-', yerr=sem_N_CCN*2, color = line_color, capsize=4, zorder=1)
        # axs[2].plot(ss_array, average_N_CCN, 'o-', color = line_color)
        axs[2].set_xlabel('Supersaturation (%)')
        axs[2].set_ylabel('CCN Concentration (cm$^{-3}$)')
        # if sem_N_CCN is not None:
        #     axs[2].fill_between(ss_array, average_N_CCN-sem_N_CCN, average_N_CCN+sem_N_CCN, color = 'gray', alpha = 0.5)

    if all_inp_temps is not None:
        axs[3].plot(all_inp_temps, all_inp_conc, c = line_color)
        axs[3].set_xbound(all_inp_temps[0], all_inp_temps[-1])
        axs[3].set_xlabel('Temperature ($\u00B0$C)')
        axs[3].set_ylabel('INP Concentration (L$^{-1}$)')

    # axs[0].set_ylim(0, 2E4)
    # axs[1].set_ylim(5E-8, 2E4)
    # axs[2].set_ylim(0, 5E3)
    # axs[3].set_ylim(0, 0.25)

    # axs[0].text(0.88, 0.97, '(a)', transform=axs[0].transAxes, fontsize=12, fontweight='bold', va='top', color='k')
    # axs[1].text(0.88, 0.97, '(b)', transform=axs[1].transAxes, fontsize=12, fontweight='bold', va='top', color='k')
    # axs[2].text(0.02, 0.97, '(c)', transform=axs[2].transAxes, fontsize=12, fontweight='bold', va='top', color='k')
    # axs[3].text(0.88, 0.97, '(d)', transform=axs[3].transAxes, fontsize=12, fontweight='bold', va='top', color='k')
    plt.tight_layout()
    plt.constrained_layout=True  # This can help further with layout constraints

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_aerosol_comprehensive.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    
    print(f'{start_time} to {end_time} aerosol comprehensive')
    if showplot:
        plt.show()

    return fig, axs


def calculate_kappa_value(start_time, end_time,
                          merged_sizes, mean_dndlogdp, ss_array, CCN_at_SS,
                          figsize=(4,4), fig = None, axs = None,
                          savefig = False, showplot = False,
                          output_folder = '', fig_name = None):
    '''
        merged_sizes: in nm
        mean_dndlogdp: in #/cc nm

        axs is a single ax
    '''
    non_nan_indices = np.where(~np.isnan(CCN_at_SS))[0]

    critical_diameters = kappa_kohler_theory.calculate_critical_diameter_interpolated(merged_sizes, mean_dndlogdp, ss_array[non_nan_indices], CCN_at_SS[non_nan_indices])
    #kappa             = kappa_kohler_theory.kappa_petter_and_Kreidenweis_2010_EQ10(critical_diameters*1E-9, ss_array[non_nan_indices]/100+1)
    kappa              = kappa_kohler_theory.calculate_kappa_fitting(critical_diameters*1E-9, ss_array[non_nan_indices]/100+1)

    

    average_kappa = np.nanmean(kappa)
    gmean_kappa   = scipy.stats.gmean(kappa) 
    gstd_kappa    = scipy.stats.gstd(kappa) # Sample Geometric Standard Deviation


    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    kappa_kohler_theory.plot_Sc_Dd_base(fig = fig, axis = axs)
    ddry   = np.logspace(1, 3, 20)  # 10 nm to 1000 nm
    Dw_k, Sc_k     = kappa_kohler_theory.find_peak_S_D_binary_search(ddry*1E-9, gmean_kappa)
    Dw_k_1, Sc_k_1 = kappa_kohler_theory.find_peak_S_D_binary_search(ddry*1E-9, gmean_kappa*gstd_kappa)
    Dw_k_2, Sc_k_2 = kappa_kohler_theory.find_peak_S_D_binary_search(ddry*1E-9, gmean_kappa/gstd_kappa)
    axs.plot(ddry*1E-3, (Sc_k-1)*100, c = '#1f77b4', ls = ':', label = rf'$\kappa={gmean_kappa:.2f} \times {gstd_kappa:.2f}^{{\pm1}}$')
    axs.fill_between(ddry*1E-3, (Sc_k_1-1)*100, (Sc_k_2-1)*100, color= '#1f77b4', alpha=0.13)
    axs.scatter(critical_diameters*1E-3, ss_array[non_nan_indices], s = 25)
    axs.legend()
    plt.tight_layout()

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_kappa.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    print(f'{start_time} to {end_time} calculate kappa')


    if showplot:
        plt.show()

    return gmean_kappa, gstd_kappa, kappa, fig, axs

def read_ACSM_kappa(start_time, end_time, acsm_kappa_file):
    """
    Compute the weighted geometric mean of ACSM κ between start_time and end_time.

    Parameters
    ----------
    start_time : str, datetime-like, or np.datetime64
        Start of the interval.
    end_time : str, datetime-like, or np.datetime64
        End of the interval.
    acsm_kappa_file : str
        Path to the ACSM kappa CSV (with columns 't_base','kappa').

    Returns
    -------
    float
        Weighted geometric mean κ over the interval (NaN if no data overlap).
    """
    # 1) Load and index
    df = pd.read_csv(acsm_kappa_file, parse_dates=['t_base'])
    df.set_index('t_base', inplace=True)

    # 2) Parse window (pd.to_datetime handles np.datetime64 directly)
    start = pd.to_datetime(start_time)
    end   = pd.to_datetime(end_time)

    # 3) Restrict to candidate hourly bins
    lower = start.floor('h')
    upper = end.floor('h')
    sub   = df.loc[lower:upper]

    if sub.empty:
        return np.nan

    # 4) Compute bin start/end arrays
    bin_starts = sub.index
    bin_ends   = bin_starts + pd.Timedelta(hours=1)

    # 5) Vectorized overlap (in hours)
    overlap = (
        np.minimum(bin_ends.values.astype('datetime64[ns]'),
                   np.datetime64(end)) -
        np.maximum(bin_starts.values.astype('datetime64[ns]'),
                   np.datetime64(start))
    ) / np.timedelta64(1, 'h')

    # 6) Mask for positive overlap and finite κ
    kappas = sub['kappa'].values
    mask   = (overlap > 0) & np.isfinite(kappas)
    if not np.any(mask):
        return np.nan

    # 7) Weighted geometric mean: exp(sum(w·ln(κ)) / sum(w))
    weights = overlap[mask]
    gm = np.exp(np.sum(weights * np.log(kappas[mask])) / np.sum(weights))

    return gm


def read_HTDMA_kappa_and_plot(start_time, end_time, all_kappa_file,
                               sizes=[50, 100, 150, 200, 250],
                               fig=None, axs=None, showplot=False,
                               savefig=False, fig_name=None, output_folder=''):
    """
    Compute κ at each dry size, then calculate overall κ. Plot both actual size-specific
    Sc points and the modeled Sc-Ddry curve from overall κ.

    Returns
    -------
    gmean_kappa : float
    gstd_kappa  : float
    fig, axs    : matplotlib objects
    """

    # Load and index data
    df = pd.read_csv(all_kappa_file, parse_dates=['hour'])
    df.set_index('hour', inplace=True)

    # Time window
    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)
    bin_starts = df.index
    bin_ends = bin_starts + pd.Timedelta(hours=1)
    overlap = (
        (np.minimum(bin_ends.values.astype('datetime64[ns]'), np.datetime64(end)) -
         np.maximum(bin_starts.values.astype('datetime64[ns]'), np.datetime64(start)))
        / np.timedelta64(1, 'h')
    )

    # Arrays for κ and Sc per size
    valid_sizes = []
    kappa_per_size = []
    gstd_per_size = []
    sc_per_size = []

    all_logs = []
    all_weights = []

    for size in sizes:
        col = f'HTDMA_kappa_at_{size}nm'
        if col not in df.columns:
            continue

        kappa_vals = df[col].values
        mask = (overlap > 0) & np.isfinite(kappa_vals)
        if not np.any(mask):
            continue

        logs = np.log(kappa_vals[mask])
        weights = overlap[mask]

        log_mean = np.sum(weights * logs) / np.sum(weights)
        log_var = np.sum(weights * (logs - log_mean)**2) / np.sum(weights)

        gmean = np.exp(log_mean)
        gstd = np.exp(np.sqrt(log_var))

        dp_m = size * 1e-9
        _, Sc = kappa_kohler_theory.find_peak_S_D_binary_search(np.array([dp_m]), gmean)

        valid_sizes.append(size)
        kappa_per_size.append(gmean)
        gstd_per_size.append(gstd)
        sc_per_size.append((Sc[0] - 1) * 100)

        all_logs.append(logs)
        all_weights.append(weights)

    if not all_logs:
        print("No valid HTDMA kappa data found in the given time window.")
        return np.nan, np.nan

    # Compute overall κ
    logs_all = np.concatenate(all_logs)
    weights_all = np.concatenate(all_weights)
    log_mean_all = np.sum(weights_all * logs_all) / np.sum(weights_all)
    log_var_all = np.sum(weights_all * (logs_all - log_mean_all)**2) / np.sum(weights_all)

    gmean_kappa = np.exp(log_mean_all)
    gstd_kappa = np.exp(np.sqrt(log_var_all))


    # Ensure fig/axs
    if fig is None or axs is None:
        fig, axs = plt.subplots(figsize=(4, 4))

    # Plot base
    kappa_kohler_theory.plot_Sc_Dd_base(fig=fig, axis=axs)

    # Plot modeled Sc–Ddry curve from overall κ
    ddry = np.logspace(1, 3, 50)  # 10 nm to 1000 nm
    D_m = ddry * 1e-9
    _, Sc_mean = kappa_kohler_theory.find_peak_S_D_binary_search(D_m, gmean_kappa)
    _, Sc_lo = kappa_kohler_theory.find_peak_S_D_binary_search(D_m, gmean_kappa * gstd_kappa)
    _, Sc_hi = kappa_kohler_theory.find_peak_S_D_binary_search(D_m, gmean_kappa / gstd_kappa)

    axs.plot(ddry * 1e-3, (Sc_mean - 1) * 100, 'b--',
             label=rf'$\kappa={gmean_kappa:.2f} \times {gstd_kappa:.2f}^{{\pm1}}$')
    axs.fill_between(ddry * 1e-3, (Sc_lo - 1) * 100, (Sc_hi - 1) * 100, color='blue', alpha=0.2)

    # Plot per-size κ points
    valid_sizes_um = np.array(valid_sizes) * 1E-3  # convert nm to μm
    axs.scatter(valid_sizes_um, sc_per_size, color='black', label='Size-specific HTDMA κ')

    print(sc_per_size)
    print(valid_sizes)

    axs.set_xlabel("Dry Diameter (μm)")
    axs.set_ylabel("Critical Supersaturation (%)")
    axs.legend()
    plt.tight_layout()

    if savefig:
        if fig_name is None:
            fig_name = f'{start.strftime("%Y%m%d_%H%M")}_{end.strftime("%Y%m%d_%H%M")}_htdma_kappa_plot.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    if showplot:
        plt.show()

    return gmean_kappa, gstd_kappa


def weighted_geometric_mean(data: np.ndarray, weights: np.ndarray, axis=0) -> np.ndarray:
    """
    Compute exp( Σ weights * ln(data) / Σ weights ) along given axis,
    correctly handling NaNs in data by zeroing corresponding weights.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        valid_mask = np.isfinite(data)
        safe_weights = np.where(valid_mask, weights, 0.0)
        safe_data = np.where(valid_mask, data, 1.0)  # log(1) = 0 → neutral for product

        numerator = np.nansum(np.log(safe_data) * safe_weights, axis=axis)
        denominator = np.nansum(safe_weights, axis=axis)

        return np.exp(numerator / denominator)


def read_HTDMA_kappa_and_plot2(start_time, end_time, all_kappa_file,
                               fig=None, axs=None, showplot=False,
                               savefig=False, fig_name=None, output_folder=''):
    
    htdma_files = retrieval_aux.filter_doearm_filenames_by_datetime(ARM_HTDMA_folder, start_time, end_time, take_previous_and_after=True)

    df_kappa_all = []  # collect each file's 2D DataFrame here
    for file in htdma_files:
        nc_ds = nc.Dataset(file, mode='r')  # Use 'r' for read-only
        # Extract variables
        base_time = nc_ds.variables['base_time'][0]
        time_offset = nc_ds.variables['time_offset'][:]
        timestamps_htdma = pd.to_datetime(base_time + time_offset, unit='s')

        raw_htdma_kappa = nc_ds.variables['kappa'][:]
        raw_htdma_conc = nc_ds.variables['aerosol_concentration'][:]
        dry_diameter_settings = nc_ds.variables['dry_diameter_setting'][:]
        nc_ds.close()

        # Apply mask
        valid_mask = ((raw_htdma_kappa > 0) & (raw_htdma_kappa < 10) & (raw_htdma_conc > 0))
        masked_kappa = np.where(valid_mask, raw_htdma_kappa, np.nan)
        masked_conc = np.where(valid_mask, raw_htdma_conc, np.nan)

        # Geometric mean per timestamp
        htdma_kappa_eff_timeseries = weighted_geometric_mean(masked_kappa, masked_conc, axis=1)
        unique_diameters = np.unique(dry_diameter_settings)
        timestamps = pd.to_datetime(base_time + time_offset, unit='s')

        # Create an empty DataFrame to store κ time series (rows: time, columns: diameter)
        df_kappa_2d = pd.DataFrame(np.nan, index=timestamps, columns=unique_diameters, dtype=np.float32)

        for t, d, kappa in zip(timestamps, dry_diameter_settings, htdma_kappa_eff_timeseries):
            df_kappa_2d.at[t, d] = kappa

        df_kappa_all.append(df_kappa_2d)

    # Concatenate all by time
    df_all_kappa = pd.concat(df_kappa_all, axis=0).sort_index()

    # Restrict to desired time range
    df_selected = df_all_kappa.loc[(df_all_kappa.index >= start_time) & (df_all_kappa.index <= end_time)]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_selected)

    # Compute per-size weighted geometric mean
    kappa_per_size = np.exp(np.nanmean(np.log(df_selected.values), axis=0))


    # Filter valid diameters
    valid_diameters = df_selected.columns[np.isfinite(kappa_per_size)].values
    kappa_per_size  = kappa_per_size[np.isfinite(kappa_per_size)]

    # Overall κ stats
    log_k = np.log(kappa_per_size)
    gmean_kappa = np.exp(log_k.mean())
    gstd_kappa  = np.exp(log_k.std())

    # Compute Sc per size
    D_m = valid_diameters * 1e-9  # m
    
    sc_per_size = []

    for D_dry, kappa in zip(D_m, kappa_per_size):
        _, Sc = kappa_kohler_theory.find_peak_S_D_binary_search(np.array([D_dry]), kappa)
        sc_per_size.append(Sc[0])

    sc_per_size = np.array(sc_per_size)
    print('valid_diameters')
    print(valid_diameters)
    print('kappa')
    print(kappa_per_size)
    print('sc')
    print(sc_per_size)

    # --- Plot ---
    if fig is None or axs is None:
        fig, axs = plt.subplots(figsize=(4, 4))

    kappa_kohler_theory.plot_Sc_Dd_base(fig=fig, axis=axs)

    # Model lines
    ddry = np.logspace(1, 3, 50)  # 10 nm to 1000 nm
    D_m_model = ddry * 1e-9
    _, Sc_mean = kappa_kohler_theory.find_peak_S_D_binary_search(D_m_model, gmean_kappa)
    _, Sc_lo   = kappa_kohler_theory.find_peak_S_D_binary_search(D_m_model, gmean_kappa * gstd_kappa)
    _, Sc_hi   = kappa_kohler_theory.find_peak_S_D_binary_search(D_m_model, gmean_kappa / gstd_kappa)

    axs.plot(ddry * 1e-3, (Sc_mean - 1) * 100, 'b--',
             label=rf'$\kappa={gmean_kappa:.2f} \times {gstd_kappa:.2f}^{{\pm1}}$')
    axs.fill_between(ddry * 1e-3, (Sc_lo - 1) * 100, (Sc_hi - 1) * 100, color='blue', alpha=0.2)

    axs.scatter(valid_diameters * 1e-3, (sc_per_size - 1) * 100, color='black', s=5, label='Size-specific HTDMA κ')
    axs.set_ylim(0.001, 0.8)

    axs.set_xlabel("Dry Diameter (μm)")
    axs.set_ylabel("Critical Supersaturation (%)")
    axs.legend()
    plt.tight_layout()

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time.strftime("%Y%m%d_%H%M")}_{end_time.strftime("%Y%m%d_%H%M")}_htdma_kappa_plot.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    if showplot:
        plt.show()

    return gmean_kappa, gstd_kappa, valid_diameters, kappa_per_size




def calculate_humidification_factor(start_time, end_time, 
                                    gmean_kappa, gstd_kappa,
                                    merged_sizes, mean_dndlogdp,
                                    rh_lim = 0.99, rh_dry = 0.3,
                                    wavelength = 532, ri_real = [1.35, 1.45, 1.50, 1.55], ri_imaginary = 0,
                                    figsize=(4,4), fig = None, axs = None,
                                    savefig = False, showplot = False, 
                                    output_folder = '', fig_name = None):
    '''
    axs is a single ax
    '''
    
    rhs = np.arange(0, 1, 0.005) 

    if isinstance(ri_real, list):
        pass
    else:
        ri_real = [ri_real]

    i_dry = np.where(np.isclose(rhs, rh_dry))[0][0]


    def hf_aux(kappa):
        all_h_factors = []
        for a_rh in rhs:
            h_factors_per_ri = []
            for a_ri_real in ri_real:
                ext_hf, sca_hf, bck_hf = kappa_kohler_theory.calculate_humidification_factor(
                    merged_sizes, mean_dndlogdp,
                    a_rh, kappa, wavelength,
                    a_ri_real, ri_imaginary
                )
                h_factors_per_ri.append(ext_hf)
            all_h_factors.append(h_factors_per_ri)

        all_h_factors = np.array(all_h_factors)  # shape: (n_RH, n_RI)

        # Normalize all curves by the dry value per RI
        for j in range(all_h_factors.shape[1]):
            dry_value = all_h_factors[i_dry, j]
            all_h_factors[:i_dry + 1, j] = dry_value  # flatten RH below/at dry
            all_h_factors[:, j] /= dry_value          # normalize entire RI curve

        # Compute min, max, mean across RIs for each RH
        hf_mean = np.nanmean(all_h_factors, axis=1)
        hf_min  = np.nanmin(all_h_factors, axis=1)
        hf_max  = np.nanmax(all_h_factors, axis=1)

        # Extra safety: ensure values at/below dry are flat and equal to 1.0
        hf_mean[:i_dry + 1] = 1.0
        hf_min[:i_dry + 1] = 1.0
        hf_max[:i_dry + 1] = 1.0

        # Build interpolators
        hf_mean_interp = scipy.interpolate.CubicSpline(rhs, hf_mean)
        hf_max_interp  = scipy.interpolate.CubicSpline(rhs, hf_max)
        hf_min_interp  = scipy.interpolate.CubicSpline(rhs, hf_min)

        return hf_mean_interp, hf_max_interp, hf_min_interp, hf_mean, hf_max, hf_min

    hf_interp, _, _, h_factors_mean, _, _        = hf_aux(gmean_kappa)
    _, hf_high_interp, _, _, h_factors_high, _   = hf_aux(gmean_kappa*gstd_kappa)
    _, _, hf_lower_interp, _, _, h_factors_low   = hf_aux(gmean_kappa/gstd_kappa)
        

    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    
    axs.plot(rhs, h_factors_mean, lw = 2, c = 'k', label = 'Humidification factor')
    axs.fill_between(rhs, h_factors_high, h_factors_low, color= 'r', alpha=0.3)
    axs.set_xbound(0, rh_lim)
    axs.set_ylim(1, hf_high_interp(rh_lim))
    axs.set_ylim(1, 12)
    #axs.legend()
    axs.set_xlabel('Relative humidity')
    axs.set_ylabel('Lidar hygroscopic \ngrowth correction factor')
    axs.xaxis.set_major_formatter(PercentFormatter(1))  # The 1 here indicates that 1.0 should be displayed as 100%

    fig.tight_layout(pad=2.0, w_pad=1.0)  # 'w_pad' controls horizontal padding

    axs.set_xticks(np.append(axs.get_xticks()[:-1], rh_lim))
    

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_humidification_factor.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    if showplot:
        plt.show()
    
    return hf_interp, hf_high_interp, hf_lower_interp, fig, axs


def calculate_bckscatter_coefficient(start_time, end_time, 
                                     merged_sizes, mean_dndlogdp,
                                     wavelength = 532, ri_real = [1.25, 1.35, 1.45], ri_imaginary = 0,
                                     fig = None, axs = None,
                                     savefig = False, showplot = False, 
                                     output_folder = '', fig_name = None):
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(ri_real)))  # generate a color for each refractive index


    for i, a_ri in enumerate(ri_real):
        complex_ri  = a_ri  - ri_imaginary * 1j
        print(f"complex_ri {complex_ri}")
        extinction_coeff, scattering_coeff, bckscatter_coeff \
                            = kappa_kohler_theory.calculate_coefficients(merged_sizes, mean_dndlogdp, complex_ri, wavelength)
        
        extinction_coeff = extinction_coeff*1E-9
        scattering_coeff = scattering_coeff*1E-9
        bckscatter_coeff = bckscatter_coeff*1E-9

        print(f"bckscatter_coeff: {bckscatter_coeff}")
        axs.scatter(bckscatter_coeff, 0, c=[colors[i]], label=f'n={a_ri}')
    
    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_backscatter_retrieval.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    if showplot:
        plt.show()

def mpl_profile_prep(start_time, end_time, 
                     cloud_free_column, 
                     mpl_range, copol_nrb, crosspol_nrb, depol,
                     nb_factor=0.1, merge_range = 2.5, wavelet = "bior3.5",
                     figsize=(6,4), fig = None, axs = None,
                     ylim = 10,
                     nrb_color = 'k', smoothed_color = 'tab:orange', depol_color = 'k',
                     savefig = False, showplot = False, 
                     output_folder = '', fig_name = None):
    '''
        axs needs to be shappe (2,)
    '''
    
    all_nrb = copol_nrb + 2 * crosspol_nrb
    all_nrb = copol_nrb

    cloudfree_nrb = all_nrb[np.where(cloud_free_column==1)[0]]
    cloudfree_nrb_mean = np.nanmean(cloudfree_nrb, axis = 0)

    print(f"number of cloud free nrb profile {cloudfree_nrb.shape}")

    cloudfree_depol_mean = np.nanmean(depol, axis = 0)
    cloudfree_depol_mean[cloudfree_depol_mean<=0] = np.nan

    #print(cloudfree_nrb_mean)

    normalized_nrb = cloudfree_nrb_mean / cloudfree_nrb_mean[0]

    nrb_smoothed = inversempl.signal_smoothing(normalized_nrb, mpl_range,
                                               nb_factor=nb_factor, wavelet = wavelet,
                                               merge_range = merge_range, plot_bool = False)
    
    
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    #print(normalized_nrb)

    axs[0].plot(normalized_nrb, mpl_range, label = 'NRB', color = nrb_color)
    axs[0].plot(nrb_smoothed, mpl_range, '--', label = 'Smoothed\nNRB', color = smoothed_color)
    axs[0].set_ylim((0, ylim))
    axs[0].set_xlim(-0.05, 1.1)
    axs[0].set_xlabel('Normalized Relative Backscatter')
    axs[0].set_ylabel('Altitude AGL (km)')
    axs[0].legend(loc='upper right')

    #axs[1].scatter(cloudfree_depol_mean, mpl_range, label = 'NRB', marker = '.', s=1, color = 'k')
    axs[1].plot(cloudfree_depol_mean, mpl_range, label = 'depol ratio', lw=1, color = depol_color)
    axs[1].set_ylim((0, ylim))
    axs[1].set_xscale('log')
    axs[1].set_ylabel('Altitude AGL (km)')
    axs[1].set_xlabel('Linear Depolarization Ratio')

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_MPL_profile.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    print(f'{start_time} to {end_time} preparing NRB and Depol profile')
    #print(f'{}')
    if showplot:
        plt.show()

    return normalized_nrb, nrb_smoothed, cloudfree_depol_mean, fig, axs


def uncertainty_analysis(coeff):
        mean_coeff = np.nanmean(coeff, axis=0)
        max_coeff  = np.nanmax(coeff, axis=0)
        min_coeff  = np.nanmin(coeff, axis=0)
        negative_uncertainty = mean_coeff - min_coeff
        positive_uncertainty = max_coeff - mean_coeff
        return (mean_coeff, negative_uncertainty, positive_uncertainty)


def retrieve_backscatter(start_time, end_time, 
                         mpl_range, nrb_smoothed, beta2,
                         rh_interpolator,
                         hf_interp, hf_high_interp, hf_low_interp,
                         calibration_range = 8, blind_zone = 0.1, RH_upper_limit = 95,
                         aerosol_lr = [30, 80],
                         calibration_beta1_ratio = [0],
                         rh_smooth_weight = 1E3, ashf_smooth_factor = False,
                         figsize=(6.5,5.5), fig = None, axs = None,
                         savefig = False, showplot = False, 
                         output_folder = '', fig_name = None): 
    
    bck_results = []

    for a_lr in aerosol_lr:
        #print(a_lr)
        for a_beta1_ratio in calibration_beta1_ratio:
            bcksca_coeff, beta2_output, inv_range = inversempl.Fernald_inversion_inwards(
                nrb_smoothed, mpl_range, beta2, S1=a_lr,
                calibration_beta1=np.interp(calibration_range, mpl_range, beta2) * a_beta1_ratio, 
                calibration_range=calibration_range)
            aerosol_bcksca_coeff = bcksca_coeff - beta2_output
            aerosol_extinc_coeff = aerosol_bcksca_coeff * a_lr
            aerosol_bcksca_coeff[aerosol_bcksca_coeff<0] = 0
            aerosol_extinc_coeff[aerosol_extinc_coeff<0] = 0
            bck_results.append({
                'calibration_range': calibration_range,
                'aerosol_lr': a_lr,
                'backscatter_coeff': bcksca_coeff,
                'beta2_output': beta2_output,
                'inversion_range': inv_range,
                'aerosol_bcksca_coeff': aerosol_bcksca_coeff,
                'aerosol_extinc_coeff': aerosol_extinc_coeff,
                'calibration_beta1_ratio': a_beta1_ratio,
            })

        
    # Extend and calculate average of all profiles
    backscatter_coeffs         = np.array([a_result["backscatter_coeff"] for a_result in bck_results])
    aerosol_backscatter_coeffs = np.array([a_result["aerosol_bcksca_coeff"] for a_result in bck_results])

    rh_profile = rh_interpolator(inv_range)
    if rh_smooth_weight:
        smotothed_rh_profile = scipy.interpolate.splev(inv_range, scipy.interpolate.splrep(inv_range, rh_profile, task = 0, s = rh_smooth_weight))
    else:
        smotothed_rh_profile = rh_profile

    rh_too_big = np.where(smotothed_rh_profile>RH_upper_limit)[0] # When Radiosonde was launched through a cloud

    ASHF_avg_profile  = hf_interp(smotothed_rh_profile/100)
    ASHF_high_profile = hf_high_interp(smotothed_rh_profile/100)
    ASHF_low_profile  = hf_low_interp(smotothed_rh_profile/100)

    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import windows, convolve

    def fatten_peaks_gaussian(profile, sigma):
        """
        Apply a 1D Gaussian filter `passes` times to flatten narrow peaks.
        Larger sigma or more passes → wider, flatter peaks.
        """
        out = profile.copy()
        out = gaussian_filter1d(out, sigma=sigma, mode='nearest')
        return out
    
    def flat_top_filter(profile, window_len):
        """
        Convolve with a flat‐top window of length window_len.
        Peaks narrower than window_len become flat‐topped.
        """
        w = max(1, int(window_len))
        if w % 2 == 0:
            w += 1
        kern = windows.flattop(w)      # flat‐top window
        kern = kern / kern.sum()       # normalize area to 1
        return convolve(profile, kern, mode='same')

    # --- in your code ---
    if ashf_smooth_factor:
        w = max(1, int(ashf_smooth_factor))
        # Option A: Flat-top window
        # ASHF_avg_profile  = flat_top_filter(ASHF_avg_profile,  window_len=w)
        # ASHF_high_profile = flat_top_filter(ASHF_high_profile, window_len=w)
        # ASHF_low_profile  = flat_top_filter(ASHF_low_profile,  window_len=w)

        # Option B: Gaussian
        ASHF_avg_profile  = fatten_peaks_gaussian(ASHF_avg_profile,  sigma=w/2)
        ASHF_high_profile = fatten_peaks_gaussian(ASHF_high_profile, sigma=w/2)
        ASHF_low_profile  = fatten_peaks_gaussian(ASHF_low_profile,  sigma=w/2)

    ASHF_avg_profile[rh_too_big] = np.nan # control the RH that gets too big
    ASHF_high_profile[rh_too_big] = np.nan
    ASHF_low_profile[rh_too_big] = np.nan

    ASHF_profiles = np.array([ASHF_avg_profile, ASHF_high_profile, ASHF_low_profile])

    dry_aerosol_backscatter_coeffs = []
    for an_aerosol_backscatter_coeff in aerosol_backscatter_coeffs:
        # Loop through each ASHF profile
        for ashf_profile in ASHF_profiles:
            # Adjust the backscatter profile by the ASHF profile
            a_dry_aerosol_backscatter_coeff = an_aerosol_backscatter_coeff / ashf_profile
            dry_aerosol_backscatter_coeffs.append(a_dry_aerosol_backscatter_coeff)
    print(f'number of dry_aerosol_backscatter_coeffs {len(dry_aerosol_backscatter_coeffs)}')        
    dry_aerosol_backscatter_coeffs = np.array(dry_aerosol_backscatter_coeffs)

    # # extend to surface
    # dry_polyfit_coefficients_list = []

    # for a_dry_bckprofile in dry_aerosol_backscatter_coeffs:
    #     fit_non_nan_indices = np.where(np.logical_and(~np.isnan(a_dry_bckprofile), inv_range < blind_zone+0.75))[0]
    #     a_coefficients = np.polyfit(inv_range[fit_non_nan_indices], a_dry_bckprofile[fit_non_nan_indices], 2)
    #     dry_polyfit_coefficients_list.append(a_coefficients)


    # blind_profiles = []
    # blind_resolution = 0.01
    # blind_inv_range = np.arange(0.01, blind_zone+blind_resolution, blind_resolution)
    # for a_coefficient in dry_polyfit_coefficients_list:
    #     a_blind_profile = np.poly1d(a_coefficient)(blind_inv_range)
    #     # a_blind_profile = a_coefficient(blind_inv_range) # for cubic spline
    #     blind_profiles.append(a_blind_profile)
    # blind_prodile_mae = uncertainty_analysis(blind_profiles)

    ###########################################
    # Define the quadratic function
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c

    # Initialize list to store polynomial coefficients for each profile
    dry_polyfit_coefficients_list = []

    # Loop over each dry backscatter profile and fit with bounds, setting min_y_intercept dynamically
    for a_dry_bckprofile in dry_aerosol_backscatter_coeffs:
        # Identify indices of valid (non-NaN) values in the fitting range
        fit_non_nan_indices = np.where(np.logical_and(~np.isnan(a_dry_bckprofile), inv_range < blind_zone + 0.5))[0]
        
        # Extract x and y data for fitting
        x_data = inv_range[fit_non_nan_indices]
        y_data = a_dry_bckprofile[fit_non_nan_indices]
        
        # Set min_y_intercept based on the value at the very bottom of the profile (first valid point)
        min_y_intercept = y_data[0] if len(y_data) > 0 else 0  # Use 0 as a fallback if no data is available
        
        # Define bounds for the fit, constraining the y-intercept (c) to be at least min_y_intercept
        bounds_lower = [-np.inf, -np.inf, min_y_intercept]
        bounds_upper = [np.inf, np.inf, np.inf]
        
        # Fit the data using curve_fit with bounds on the coefficients
        popt, _ = curve_fit(
            quadratic, x_data, y_data, 
            bounds=(bounds_lower, bounds_upper)
        )
        
        # Append the optimized coefficients to the list
        dry_polyfit_coefficients_list.append(popt)

    # Generate profiles for the blind range using the constrained polynomials
    blind_profiles = []
    blind_resolution = 0.01
    blind_inv_range = np.arange(0.01, blind_zone + blind_resolution, blind_resolution)

    for a_coefficient in dry_polyfit_coefficients_list:
        # Generate profile in the blind range using the fitted polynomial coefficients
        a_blind_profile = quadratic(blind_inv_range, *a_coefficient)
        blind_profiles.append(a_blind_profile)

    # Calculate the uncertainty metric (e.g., MAE) for the blind profiles
    blind_profile_mae = uncertainty_analysis(blind_profiles)
    ###########################################

    
    # MEAN AND ERRORS
    bcksca_coeff_mae       = uncertainty_analysis(backscatter_coeffs)
    aerosol_bcksca_mae     = uncertainty_analysis(aerosol_backscatter_coeffs)
    dry_aerosol_bcksca_mae = uncertainty_analysis(dry_aerosol_backscatter_coeffs)

    # Plotting
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    color_cycle = cycle(plt.cm.tab20(np.linspace(0, 1, 8)))  # Uses a colormap to generate many colors
    ls_cycle    = cycle(['-', '--', ':', '-.'])  # Basic line styles, cycling repeats these



    
    
    axs[0].plot(beta2_output, inv_range, label = f'$\\beta_2$', color = 'b', ls = '--', lw = 2)

    axs[0].plot(bcksca_coeff_mae[0], inv_range, color = 'k', lw = 1.5, label = '$\\beta_1+\\beta_2$')
    axs[0].fill_betweenx(inv_range, bcksca_coeff_mae[0]-bcksca_coeff_mae[1], bcksca_coeff_mae[0]+bcksca_coeff_mae[2], color = 'gray', alpha=0.4)

    #axs[0].plot(avg_bcksca_coeff, max_inv_range, color = 'k', lw = 2)
    axs[1].plot(aerosol_bcksca_mae[0], inv_range, color = 'k', lw = 1.5, label = '$\\beta_1$', zorder = 1)
    axs[1].fill_betweenx(inv_range, aerosol_bcksca_mae[0]-aerosol_bcksca_mae[1], aerosol_bcksca_mae[0]+aerosol_bcksca_mae[2], color = 'gray', alpha=0.4, zorder = 1)

    axs[1].plot(dry_aerosol_bcksca_mae[0], inv_range, color = 'tomato', lw = 1.5, label = '$\\beta_{dry}}$', zorder = 2)
    #axs[1].fill_betweenx(inv_range, dry_aerosol_bcksca_mae[0]-dry_aerosol_bcksca_mae[1], dry_aerosol_bcksca_mae[0]+dry_aerosol_bcksca_mae[1], color = 'salmon', alpha=0.4, zorder = 2)

    # Blind range
    axs[1].plot(blind_profile_mae[0], blind_inv_range, color = 'tomato', lw = 1.5, ls = ':', zorder = 3)
    axs[1].fill_betweenx(np.concatenate([blind_inv_range, inv_range]), 
                         np.concatenate([blind_profile_mae[0]-blind_profile_mae[1], dry_aerosol_bcksca_mae[0]-dry_aerosol_bcksca_mae[1]]), 
                         np.concatenate([blind_profile_mae[0]+blind_profile_mae[2], dry_aerosol_bcksca_mae[0]+dry_aerosol_bcksca_mae[2]]), color = 'salmon', alpha=0.4, zorder = 2)
    axs[1].plot(blind_profile_mae[0]-blind_profile_mae[1], blind_inv_range, color = 'tomato', lw = 1.5, ls = ':', zorder = 3)
    axs[1].plot(blind_profile_mae[0]+blind_profile_mae[2], blind_inv_range, color = 'tomato', lw = 1.5, ls = ':', zorder = 3)

    ax_rh = axs[1].twiny()
    ax_rh.set_xlabel('RH %')
    #ax_rh.plot(rh_profile, inv_range, c='b', ls=':', alpha = 0.3)
    ax_rh.plot(smotothed_rh_profile, inv_range, c='b', ls='-', label = 'RH', alpha = 0.6, zorder = 0)
    ax_rh.set_xlim(0, 100)
    

    handles, labels = [], []
    for ax in [axs[1], ax_rh]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    axs1_legend = ax_rh.legend(handles, labels, loc='upper right', framealpha=1)

    axs[0].set_ylim(0, 8)
    axs[1].set_ylim(0, 8)
    axs[1].set_xlim(0)
    axs[1].set_xlim(0)
    axs[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axs[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axs[0].set_xlabel('Backscatter Coefficient (km$^{-1}$)')
    axs[1].set_xlabel('Backscatter Coefficient (km$^{-1}$)')
    axs[0].legend()
    axs[0].set_ylabel('Altitude AGL (km)')
    axs[1].set_ylabel('Altitude AGL (km)')

    if savefig:
        if fig_name is None:
            fig_name = f'{start_time}_{end_time}_backscatter_retrieval.png'
        fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

    if showplot:
        plt.show()


    return (bck_results, inv_range, dry_aerosol_backscatter_coeffs, bcksca_coeff_mae, 
            aerosol_bcksca_mae, dry_aerosol_bcksca_mae, smotothed_rh_profile, 
            blind_inv_range, blind_profile_mae, dry_polyfit_coefficients_list, fig, axs)


def replace_zeros_with_nan(arr):
    arr = np.array(arr)
    arr[arr <= 0] = np.nan
    return arr


# def aerosol_profiles(start_time, end_time,
#                      inv_range, dry_aerosol_backscatter_coeffs,
#                      aerosol_concentration, aerosol_concentration_uncertainty,
#                      ss_array, average_N_CCN, uncertainty_N_CCN,
#                      inp_temp_list, inp_conc_list, ylim = 6,
#                      figsize=(10,5.5), fig = None, axs = None,
#                      savefig = False, showplot = False, blind_zone = 0.1, 
#                      output_folder = '', fig_name = None):

def aerosol_profiles(start_time, end_time,
                     inv_range, dry_aerosol_backscatter_coeffs,
                     aerosol_concentration, aerosol_concentration_uncertainty,
                     ss_array, average_N_CCN, uncertainty_N_CCN,
                     inp_temp_list, inp_conc_list,
                     dry_polyfit_coefficients_list, blind_inv_range,  # New parameters
                     ylim=6, figsize=(10, 5.5), fig=None, axs=None,
                     savefig=False, showplot=False, blind_zone=0.1, 
                     output_folder='', fig_name=None):
    
    
    # Use precomputed blind profiles instead of recalculating the fit
    blind_dry_bcksca_coeff_profiles = [np.poly1d(coeff)(blind_inv_range) for coeff in dry_polyfit_coefficients_list]
    aerosol_profiles = []
    blind_aerosol_profiles = []

    for i, a_dry_profile in enumerate(dry_aerosol_backscatter_coeffs):
        a_blind_dry_bcksca_profile = blind_dry_bcksca_coeff_profiles[i]
        
        for a_aerosol_concentration in [
            aerosol_concentration - aerosol_concentration_uncertainty, 
            aerosol_concentration, 
            aerosol_concentration + aerosol_concentration_uncertainty
        ]:
            aerosol_profiles.append(a_aerosol_concentration * a_dry_profile / a_blind_dry_bcksca_profile[0])
            blind_aerosol_profiles.append(a_aerosol_concentration * a_blind_dry_bcksca_profile / a_blind_dry_bcksca_profile[0])

    # Calculate uncertainties
    aerosol_profile_mae = uncertainty_analysis(aerosol_profiles)
    blind_aerosol_profile_mae = uncertainty_analysis(blind_aerosol_profiles)

    CCN_maes = [] # store data for looping through supersaturation arrays
    blind_CCN_maes = []
    for i, an_ss in enumerate(ss_array):
        CCN_profiles = []
        blind_CCN_profiles = []
        for j, a_dry_profile in enumerate(dry_aerosol_backscatter_coeffs):
            for a_N_CCN in [average_N_CCN[i]-uncertainty_N_CCN[i], average_N_CCN[i], average_N_CCN[i]+uncertainty_N_CCN[i]]:
                CCN_profiles.append(a_N_CCN * a_dry_profile/blind_dry_bcksca_coeff_profiles[j][0])
                blind_CCN_profiles.append(a_N_CCN * blind_dry_bcksca_coeff_profiles[j]/blind_dry_bcksca_coeff_profiles[j][0])
        a_CCN_mae = uncertainty_analysis(CCN_profiles)
        a_blind_CCN_mae = uncertainty_analysis(blind_CCN_profiles)

        CCN_maes.append(a_CCN_mae)
        blind_CCN_maes.append(a_blind_CCN_mae)
    
    INP_maes = []
    blind_INP_maes = []
    if inp_temp_list is not None:
        inp_colors = plt.cm.tab10(np.linspace(0, 1, len(inp_temp_list)))
        
        INP_plot_temps = []
        INP_plot_colors = []
        for i, an_temp in enumerate(inp_temp_list):
            if inp_conc_list[i] != 0:
                INP_plot_temps.append(an_temp)
                INP_plot_colors.append(inp_colors[i])
                INP_profiles = []
                blind_INP_profiles = []
                for j, a_dry_profile in enumerate(dry_aerosol_backscatter_coeffs):
                    INP_profiles.append(inp_conc_list[i] * a_dry_profile/blind_dry_bcksca_coeff_profiles[j][0])
                    blind_INP_profiles.append(inp_conc_list[i] * blind_dry_bcksca_coeff_profiles[j]/blind_dry_bcksca_coeff_profiles[j][0])
                a_INP_mae = uncertainty_analysis(INP_profiles)
                a_blind_INP_mae = uncertainty_analysis(blind_INP_profiles)
                INP_maes.append(a_INP_mae)
                blind_INP_maes.append(a_blind_INP_mae)

        if fig is None or axs is None:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    

    #axs[0].plot(blind_aerosol_profile_mae[0], blind_inv_range, color = 'k', ls = ':')
    # axs[0].plot(blind_aerosol_profile_mae[0]-blind_aerosol_profile_mae[1], blind_inv_range, color = 'k', ls = ':')
    # axs[0].plot(blind_aerosol_profile_mae[0]+blind_aerosol_profile_mae[2], blind_inv_range, color = 'k', ls = ':')
    
    ccn_colors = plt.cm.tab10(np.linspace(0, 1, len(ss_array)))
    #ccn_colors = ['tab:blue', 'tab:orange','tab:green'] # for paper
    if CCN_maes:
        for i in range(len(CCN_maes)):
            j = i
        #for j in range(3):
            #i = [0,2,5][j]
            #axs[0].plot(CCN_maes[i][0], inv_range, color=ccn_colors[i], label = f'SS={ss_array[i]:.2f}%') # not using standard error
            axs[0].plot(np.concatenate([blind_CCN_maes[i][0], CCN_maes[i][0]]), np.concatenate([blind_inv_range, inv_range]), color=ccn_colors[j], label = f'$s_{{s}}={ss_array[i]:.2f}%$')
            #axs[0].fill_betweenx(inv_range, CCN_maes[i][0]-CCN_maes[i][1], CCN_maes[i][0]+CCN_maes[i][2], color=ccn_colors[i], alpha=0.3)
            axs[0].fill_betweenx(np.concatenate([blind_inv_range, inv_range]), \
                                np.concatenate([blind_CCN_maes[i][0]-blind_CCN_maes[i][1], CCN_maes[i][0]-CCN_maes[i][1]]), \
                                np.concatenate([blind_CCN_maes[i][0]+blind_CCN_maes[i][2], CCN_maes[i][0]+CCN_maes[i][2]]), color=ccn_colors[j], alpha=0.3)
            axs[0].plot(blind_CCN_maes[i][0], blind_inv_range, color=ccn_colors[j], ls = ':')

            axs[1].plot(replace_zeros_with_nan(CCN_maes[i][0]), inv_range, color=ccn_colors[j], label = f'SS={ss_array[i]:.2f}%')
            axs[1].fill_betweenx(inv_range, replace_zeros_with_nan(CCN_maes[i][0]-CCN_maes[i][1]), replace_zeros_with_nan(CCN_maes[i][0]+CCN_maes[i][2]), color=ccn_colors[j], alpha=0.3)

    axs[0].plot(np.concatenate([blind_aerosol_profile_mae[0], aerosol_profile_mae[0]]), np.concatenate([blind_inv_range, inv_range]), color = 'k', label = 'Aerosol')
    axs[0].fill_betweenx(np.concatenate([blind_inv_range, inv_range]), \
                         np.concatenate([blind_aerosol_profile_mae[0]-blind_aerosol_profile_mae[1], aerosol_profile_mae[0]-aerosol_profile_mae[1]]), \
                         np.concatenate([blind_aerosol_profile_mae[0]+blind_aerosol_profile_mae[2], aerosol_profile_mae[0]+aerosol_profile_mae[2]]), \
                         color = 'k', alpha=0.3)
    axs[1].plot(replace_zeros_with_nan(aerosol_profile_mae[0]), inv_range, color = 'k', label = 'Aerosol', zorder = 2)
    axs[1].fill_betweenx(inv_range, replace_zeros_with_nan(aerosol_profile_mae[0]-aerosol_profile_mae[1]), replace_zeros_with_nan(aerosol_profile_mae[0]+aerosol_profile_mae[2]), color = 'k', alpha=0.3, zorder = 1)
    
    if inp_temp_list is not None:
        if INP_maes:
            for i in reversed(range(len(INP_maes))):
                if False:   # Optional Dashed Line Style
                    axs[2].plot(np.concatenate([blind_INP_maes[i][0],INP_maes[i][0]]), 
                            np.concatenate([blind_inv_range, inv_range]), 
                            color=INP_plot_colors[i], 
                            label = f'T={INP_plot_temps[i]}$^\circ$C', lw = 2.5, clip_on=False, ls = '--')
                    axs[2].fill_betweenx(np.concatenate([blind_inv_range, inv_range]), 
                                        np.concatenate([blind_INP_maes[i][0]-blind_INP_maes[i][1], INP_maes[i][0]-INP_maes[i][1]]), 
                                        np.concatenate([blind_INP_maes[i][0]+blind_INP_maes[i][2], INP_maes[i][0]+INP_maes[i][2]]), 
                                        color=INP_plot_colors[i], alpha=0.3)
                else:
                    axs[2].plot(np.concatenate([blind_INP_maes[i][0],INP_maes[i][0]]), 
                                np.concatenate([blind_inv_range, inv_range]), 
                                color=INP_plot_colors[i], 
                                label = f'T={INP_plot_temps[i]}$^\circ$C', lw = 2.5, clip_on=False)
                    axs[2].fill_betweenx(np.concatenate([blind_inv_range, inv_range]), 
                                        np.concatenate([blind_INP_maes[i][0]-blind_INP_maes[i][1], INP_maes[i][0]-INP_maes[i][1]]), 
                                        np.concatenate([blind_INP_maes[i][0]+blind_INP_maes[i][2], INP_maes[i][0]+INP_maes[i][2]]), 
                                        color=INP_plot_colors[i], alpha=0.3)

                axs[3].plot(replace_zeros_with_nan(INP_maes[i][0]), inv_range, 
                            color=INP_plot_colors[i], 
                            label = f'T={INP_plot_temps[i]}$^\circ$C', clip_on=False)
                axs[3].fill_betweenx(inv_range, 
                                    replace_zeros_with_nan(INP_maes[i][0]-INP_maes[i][1]), 
                                    replace_zeros_with_nan(INP_maes[i][0]+INP_maes[i][2]), 
                                    color=INP_plot_colors[i], alpha=0.3)
    
    axs[0].set_xlabel('Concentration (cm$^-$$^3$)')
    axs[0].set_ylabel('Altitude AGL (km)')
    axs[0].set_ylim(0, 8)
    axs[0].legend()
    axs[0].set_xlim(0)
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Concentration (cm$^-$$^3$)')
    axs[1].set_ylabel('Altitude AGL (km)')
    axs[1].set_ylim(0, 8)
    #axs[2].set_xscale('log')
    axs[2].set_xlabel('Concentration (L$^-$$^1$)')
    axs[2].set_ylabel('Altitude AGL (km)')
    axs[2].set_ylim(0, 8)
    axs[2].set_xlim(0)
    # axs[2].set_xlim(0,0.0025)
    # axs[2].set_ylim(0,1.5)
    #axs[2].set_xlim(1E-5, 1)
    axs[2].legend()
    axs[0].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))
    axs[3].set_xscale('log')
    axs[3].set_xlabel('Concentration (#/L)')
    axs[3].set_ylabel('Altitude AGL (km)')
    
    axs[3].set_ylim(0, 8)
    #axs[1].ticklabel_format(axis='x', style='sci', scilimits=(-1, 1))

    if showplot:
        plt.show()

    return aerosol_profile_mae, blind_aerosol_profile_mae, CCN_maes, blind_CCN_maes, INP_maes, blind_INP_maes


def output_data(start_time, end_time, station_name,
                nrb_profile, nrb_smoothed, depol_profile, mpl_range,
                bcksca_coeff_profile, inv_range,
                aerosol_bcksca_coeff_profile, 
                dry_aerosol_bcksca_coeff_profile,
                smotothed_rh_profile,
                merged_diameter, mean_dndlogdp, 
                ss_CCN, conc_CCN, stdev_conc_CCN, ste_CCN,
                temp_INP, conc_INP,
                aerosol_concentration, ste_aerosol,
                kappa, kappa_gstdev,
                blind_inv_range, blind_profile_mae,
                aerosol_profile_mae, blind_aerosol_profile_mae, CCN_maes, blind_CCN_maes,
                output_folder):
    
    nrb_profile = np.interp(inv_range, mpl_range, nrb_profile)          # So only needs to report one range
    nrb_smoothed = np.interp(inv_range, mpl_range, nrb_smoothed)
    depol_profile = np.interp(inv_range, mpl_range, depol_profile)

    
    file_name = f'{station_name}_{start_time}_{end_time}_traceravp.nc'
    ds = nc.Dataset(os.path.join(output_folder, file_name), 'w', format='NETCDF4')

    dimension_range        = ds.createDimension('range', len(inv_range))
    dimension_aerosol_size = ds.createDimension('size',  len(merged_diameter))
    dimension_ss           = ds.createDimension('ss',    len(ss_CCN))

    dimension_blind_range  = ds.createDimension('blind_range', len(blind_inv_range))

    variable_range                   = ds.createVariable('range', 'f8', ('range',)) # inv_range
    variable_range.unit              = "km"
    variable_range.longname          = "lidar range"

    variable_blind_range             = ds.createVariable('blind_range', 'f8', ('blind_range',)) # inv_range
    variable_blind_range.unit        = "km"
    variable_blind_range.longname    = "lidar blind range"

    variable_nrb                     = ds.createVariable('NRB', 'f8', ('range',))
    variable_nrb.unit                = "unitless"
    variable_nrb.longname            = "cloud free normalized relative backscatter profile: copol_nrb + 2 \u00D7 crosspol_nrb"

    variable_smoothed_nrb            = ds.createVariable('NRB_smoothed', 'f8', ('range',))
    variable_smoothed_nrb.unit       = "unitless"
    variable_smoothed_nrb.longname   = "wavelet smoothed cloud free normalized relative backscatter profile"

    variable_depol                   = ds.createVariable('depol_ratio', 'f8', ('range',))
    variable_depol.unit              = "unitless"
    variable_depol.longname          = "linear depolarization ratio profile"

    variable_bckscacoeff             = ds.createVariable('bcksca_coeff', 'f8', ('range',))
    variable_bckscacoeff.unit        = "1/km"
    variable_bckscacoeff.longname    = "total backscatter coefficient profile: \u03B2\u2081 + \u03B2\u2082"
    variable_bckscacoeff_negative_uncertainty \
                                     = ds.createVariable('bcksca_coeff_negative_err', 'f8', ('range',))
    variable_bckscacoeff_negative_uncertainty.unit \
                                     = "1/km"
    variable_bckscacoeff_negative_uncertainty.longname \
                                     = "inferred negative uncertainty profile for total backscatter coefficient"
    variable_bckscacoeff_positive_uncertainty \
                                     = ds.createVariable('bcksca_coeff_positive_err', 'f8', ('range',))
    variable_bckscacoeff_positive_uncertainty.unit \
                                     = "1/km"
    variable_bckscacoeff_positive_uncertainty.longname \
                                     = "inferred positive uncertainty profile for total backscatter coefficient"
    

    variable_aerosol_bckscacoeff     = ds.createVariable('aerosol_bcksca_coeff', 'f8', ('range',))
    variable_aerosol_bckscacoeff.unit \
                                     = "1/km"
    variable_aerosol_bckscacoeff.longname \
                                     = "aerosol backscatter coefficient profile: \u03B2\u2081"
    variable_aerosol_bckscacoeff_negative_uncertainty \
                                     = ds.createVariable('aerosol_bcksca_coeff_negative_err', 'f8', ('range',))
    variable_aerosol_bckscacoeff_negative_uncertainty.unit \
                                     = "1/km"
    variable_aerosol_bckscacoeff_negative_uncertainty.longname \
                                     = "inferred negative uncertainty profile for aerosol backscatter coefficient"
    variable_aerosol_bckscacoeff_positive_uncertainty \
                                     = ds.createVariable('aerosol_bcksca_coeff_positive_err', 'f8', ('range',))
    variable_aerosol_bckscacoeff_positive_uncertainty.unit \
                                     = "1/km"
    variable_aerosol_bckscacoeff_positive_uncertainty.longname \
                                     = "inferred positive uncertainty profile for aerosol backscatter coefficient"
    
    variable_aerosol_dry_bckscacoeff = ds.createVariable('aerosol_dry_bcksca_coeff', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff.unit \
                                     = "1/km"
    variable_aerosol_dry_bckscacoeff.longname \
                                     = "dry aerosol backscatter coefficient profile: dry_\u03B2\u2081 = f(RH) \u00D7 \u03B2\u2081"
    variable_aerosol_dry_bckscacoeff.description \
                                     = "Use this variable to calculate aerosol, CCN, and INP vertical profile. For example, CCN(h) = CCN(h0) \u00D7 dry_\u03B2\u2081(h)/dry_\u03B2\u2081(h0)"
    variable_aerosol_dry_bckscacoeff_negative_uncertainty \
                                     = ds.createVariable('aerosol_dry_bcksca_coeff_negative_err', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff_negative_uncertainty.unit \
                                     = "1/km"
    variable_aerosol_dry_bckscacoeff_negative_uncertainty.longname \
                                     = "inferred negative uncertainty profile for dry aerosol backscatter coefficient"
    variable_aerosol_dry_bckscacoeff_positive_uncertainty \
                                     = ds.createVariable('aerosol_dry_bcksca_coeff_positive_err', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff_positive_uncertainty.unit \
                                     = "1/km"
    variable_aerosol_dry_bckscacoeff_positive_uncertainty.longname \
                                     = "inferred positive uncertainty profile for dry aerosol backscatter coefficient"
    


    variable_blind_aerosol_dry_bckscacoeff = ds.createVariable('blind_aerosol_dry_bcksca_coeff', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff.unit \
                                     = "1/km"
    variable_blind_aerosol_dry_bckscacoeff.longname \
                                     = "blind range dry aerosol backscatter coefficient profile: dry_\u03B2\u2081 = f(RH) \u00D7 \u03B2\u2081"
    variable_blind_aerosol_dry_bckscacoeff.description \
                                     = "Use this variable to calculate aerosol, CCN, and INP vertical profile. For example, CCN(h) = CCN(h0) \u00D7 dry_\u03B2\u2081(h)/dry_\u03B2\u2081(h0)"
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty \
                                     = ds.createVariable('blind_aerosol_dry_bcksca_coeff_negative_err', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty.unit \
                                     = "1/km"
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty.longname \
                                     = "blind range inferred negative uncertainty profile for dry aerosol backscatter coefficient"
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty \
                                     = ds.createVariable('blind_aerosol_dry_bcksca_coeff_positive_err', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty.unit \
                                     = "1/km"
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty.longname \
                                     = "blind range inferred positive uncertainty profile for dry aerosol backscatter coefficient"




    variable_rh_profile              = ds.createVariable('rh', 'f8', ('range',))
    variable_rh_profile.unit         = "unitless"
    variable_rh_profile.longname     = "smoothed relative humidity profile"

    variable_aerosol_diameter        = ds.createVariable('diameter', 'f8', ('size',))
    variable_aerosol_diameter.unit   = "nm"
    variable_aerosol_diameter.longname \
                                     = "aerosol diameter"
    variable_aerosol_dndlogdp        = ds.createVariable('dndlogdp', 'f8', ('size',))
    variable_aerosol_dndlogdp.unit   = "#/cm\u00B3"
    variable_aerosol_dndlogdp.longname \
                                     = "aerosol size distribution dNdlogDp"

    variable_ccn_ss                  = ds.createVariable('ccn_ss', 'f8', ('ss',))
    variable_ccn_ss.unit             = "%"
    variable_ccn_ss.longname         = "supersaturations CCN concentration is evaluated at"
    variable_ccn_conc                = ds.createVariable('ccn_conc', 'f8', ('ss',))
    variable_ccn_conc.unit           = "#/cm\u00B3"
    variable_ccn_conc.longname       = "average CCN concentration"
    variable_ccn_conc_stdev          = ds.createVariable('ccn_conc_stdev', 'f8', ('ss',))
    variable_ccn_conc_stdev.unit     = "#/cm\u00B3"
    variable_ccn_conc_stdev.longname = "CCN concentration standard deviation"

    variable_ccn_conc_ste          = ds.createVariable('ccn_conc_ste', 'f8', ('ss',))
    variable_ccn_conc_ste.unit     = "#/cm\u00B3"
    variable_ccn_conc_ste.longname = "CCN concentration standard error"

    variable_aerosol_concentration   = ds.createVariable('aerosol_concentration', 'f8')
    variable_aerosol_concentration.unit \
                                     = "#/cm\u00B3"
    variable_aerosol_concentration.longname \
                                     = "aerosol concetration calculated"
    
    variable_ste_aerosol_concentration   = ds.createVariable('aerosol_concentration_ste', 'f8')
    variable_ste_aerosol_concentration.unit \
                                     = "#/cm\u00B3"
    variable_ste_aerosol_concentration.longname \
                                     = "aerosol concetration standard error"
    
    variable_kappa                   = ds.createVariable('kappa', 'f8')
    variable_kappa.unit              = "unitless"
    variable_kappa.longname          = "hygroscopicity kappa calculated from CCN concentration and aerosol size distribution"
    variable_kappa_gstdev            = ds.createVariable('kappa_gstdev', 'f8')
    variable_kappa_gstdev.unit       = "unitless"
    variable_kappa_gstdev.longname   = "hygroscopicity kappa geometric standard deviation"


    variable_range[:]                               = inv_range
    variable_blind_range[:]                         = blind_inv_range
    variable_nrb[:]                                 = nrb_profile
    variable_smoothed_nrb[:]                        = nrb_smoothed
    variable_depol[:]                               = depol_profile
    variable_bckscacoeff[:]                         = bcksca_coeff_profile[0]
    variable_bckscacoeff_negative_uncertainty[:]    = bcksca_coeff_profile[1]
    variable_bckscacoeff_positive_uncertainty[:]    = bcksca_coeff_profile[2]
    variable_aerosol_bckscacoeff[:]                 = aerosol_bcksca_coeff_profile[0]
    variable_aerosol_bckscacoeff_negative_uncertainty[:]     = aerosol_bcksca_coeff_profile[1]
    variable_aerosol_bckscacoeff_positive_uncertainty[:]     = aerosol_bcksca_coeff_profile[2]
    variable_aerosol_dry_bckscacoeff[:]             = dry_aerosol_bcksca_coeff_profile[0]
    variable_aerosol_dry_bckscacoeff_negative_uncertainty[:] = dry_aerosol_bcksca_coeff_profile[1]
    variable_aerosol_dry_bckscacoeff_positive_uncertainty[:] = dry_aerosol_bcksca_coeff_profile[2]
    variable_blind_aerosol_dry_bckscacoeff[:]             = blind_profile_mae[0]
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty[:] = blind_profile_mae[1]
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty[:] = blind_profile_mae[2]
    variable_rh_profile[:]                          = smotothed_rh_profile
    variable_aerosol_diameter[:]                    = merged_diameter
    variable_aerosol_dndlogdp[:]                    = mean_dndlogdp
    variable_ccn_ss[:]                              = ss_CCN
    variable_ccn_conc[:]                            = conc_CCN
    variable_ccn_conc_stdev[:]                      = stdev_conc_CCN
    variable_ccn_conc_ste[:]                        = ste_CCN

    variable_aerosol_concentration.assignValue(aerosol_concentration)
    variable_ste_aerosol_concentration.assignValue(ste_aerosol)
    variable_kappa.assignValue(kappa)
    variable_kappa_gstdev.assignValue(kappa_gstdev)


    if np.any(temp_INP):
        dimension_temp                   = ds.createDimension('temp',    len(temp_INP))
        variable_inp_temp                = ds.createVariable('inp_temp', 'f8', ('temp',))
        variable_inp_temp.unit           = "\u00B0C"
        variable_inp_temp.longname       = "temperature INP concentration is evaluated at"
        variable_inp_conc                = ds.createVariable('inp_conc', 'f8', ('temp',))
        variable_inp_conc.unit           = "#/L"
        variable_inp_conc.longname       = "average INP concentration"
        variable_inp_temp[:]             = temp_INP
        variable_inp_conc[:]             = conc_INP

    ds.description = f'lidar retrieved aerosol vertical profile, aerosol size distribution, CCN spectra, and INP spectra collected at {station_name}'

    
    ds.close()

def output_data_with_lidar_details(start_time, end_time, station_name,
                                   nrb_profile, nrb_smoothed, depol_profile, mpl_range,
                                   bcksca_coeff_profile, inv_range,
                                   aerosol_bcksca_coeff_profile,
                                   dry_aerosol_bcksca_coeff_profile,
                                   smotothed_rh_profile,
                                   merged_diameter, mean_dndlogdp,
                                   ss_CCN, conc_CCN, stdev_conc_CCN, ste_CCN,
                                   temp_INP, conc_INP,
                                   aerosol_concentration, ste_aerosol,
                                   kappa, kappa_gstdev,
                                   blind_inv_range, blind_profile_mae,
                                   aerosol_profile_mae, blind_aerosol_profile_mae, CCN_maes, blind_CCN_maes,
                                   output_folder,
                                   mpl_object, cloud_mask, cloud_free_column,
                                   find_cloud_range):
    """
    Outputs data to a NetCDF file, including lidar data and other atmospheric measurements.
    Assumes that mpl_object, cloud_mask, and cloud_free_column are provided.
    """
    import numpy as np
    import netCDF4 as nc
    import os

    # Interpolate lidar profiles to inversion range if necessary
    nrb_profile_interp = np.interp(inv_range, mpl_range, nrb_profile)
    nrb_smoothed_interp = np.interp(inv_range, mpl_range, nrb_smoothed)
    depol_profile_interp = np.interp(inv_range, mpl_range, depol_profile)

    file_name = f'{station_name}_{start_time}_{end_time}_traceravp.nc'
    ds = nc.Dataset(os.path.join(output_folder, file_name), 'w', format='NETCDF4')

    # Define dimensions
    dimension_range = ds.createDimension('range', len(inv_range))
    dimension_aerosol_size = ds.createDimension('size', len(merged_diameter))
    dimension_ss = ds.createDimension('ss', len(ss_CCN))
    dimension_blind_range = ds.createDimension('blind_range', len(blind_inv_range))
    dimension_time = ds.createDimension('time', len(mpl_object.interpolated_datetime))
    dimension_mpl_range = ds.createDimension('mpl_range', len(mpl_object.range))

    # Define cloud finding range indices
    cloud_finding_range_indices = np.where(mpl_object.range < find_cloud_range)[0]
    dimension_cloud_range = ds.createDimension('cloud_range', len(cloud_finding_range_indices))

    # Variables for time and MPL range (lidar)
    variable_time = ds.createVariable('time', 'f8', ('time',))
    variable_time.units = 'seconds since 1970-01-01 00:00:00 UTC'
    variable_time.long_name = 'Interpolated MPL datetime'

    variable_mpl_range = ds.createVariable('mpl_range', 'f8', ('mpl_range',))
    variable_mpl_range.unit = 'km'
    variable_mpl_range.long_name = 'MPL range bins'

    variable_cloud_range = ds.createVariable('cloud_range', 'f8', ('cloud_range',))
    variable_cloud_range.unit = 'km'
    variable_cloud_range.long_name = 'MPL range bins for cloud detection'

    # Convert numpy.datetime64 to Unix time (seconds since epoch)
    numeric_times = mpl_object.interpolated_datetime.astype('datetime64[s]').astype('int')

    # Assign data to time and range variables
    variable_time[:] = numeric_times
    variable_mpl_range[:] = mpl_object.range
    variable_cloud_range[:] = mpl_object.range[cloud_finding_range_indices]

    # Variables for lidar data (using 'mpl_range' dimension)
    variable_interpolated_nrb_copol = ds.createVariable('interpolated_nrb_copol', 'f8', ('time', 'mpl_range'), zlib=True)
    variable_interpolated_nrb_copol.units = 'unitless'
    variable_interpolated_nrb_copol.long_name = 'Interpolated copolarized NRB'

    variable_interpolated_depol_ratio = ds.createVariable('interpolated_depol_ratio', 'f8', ('time', 'mpl_range'), zlib=True)
    variable_interpolated_depol_ratio.units = 'unitless'
    variable_interpolated_depol_ratio.long_name = 'Interpolated depolarization ratio'

    variable_interpolated_snr_copol = ds.createVariable('interpolated_snr_copol', 'f8', ('time', 'mpl_range'), zlib=True)
    variable_interpolated_snr_copol.units = 'unitless'
    variable_interpolated_snr_copol.long_name = 'Interpolated signal-to-noise ratio (copolarized)'

    # Variables for cloud mask (using 'cloud_range' dimension)
    variable_cloud_mask = ds.createVariable('cloud_mask', 'i1', ('time', 'cloud_range'), zlib=True)
    variable_cloud_mask.units = '1'
    variable_cloud_mask.long_name = 'Cloud mask (1 for cloud, 0 for clear)'

    variable_cloud_free_column = ds.createVariable('cloud_free_column', 'i1', ('time',))
    variable_cloud_free_column.units = '1'
    variable_cloud_free_column.long_name = 'Cloud-free column indicator (1 for cloud-free, 0 otherwise)'

    # Assign data to lidar variables
    variable_interpolated_nrb_copol[:, :] = mpl_object.interpolated_nrb_copol
    variable_interpolated_depol_ratio[:, :] = mpl_object.interpolated_depol_ratio
    variable_interpolated_snr_copol[:, :] = mpl_object.interpolated_snr_copol
    variable_cloud_mask[:, :] = cloud_mask[:, cloud_finding_range_indices]
    variable_cloud_free_column[:] = cloud_free_column

    # Variables for inversion range
    variable_range = ds.createVariable('range', 'f8', ('range',))  # inv_range
    variable_range.unit = "km"
    variable_range.longname = "Inversion range"

    variable_blind_range = ds.createVariable('blind_range', 'f8', ('blind_range',))
    variable_blind_range.unit = "km"
    variable_blind_range.longname = "Lidar blind range"

    # Variables for profiles (defined over 'range' dimension)
    variable_nrb = ds.createVariable('NRB', 'f8', ('range',))
    variable_nrb.unit = "unitless"
    variable_nrb.longname = "Cloud-free normalized relative backscatter profile: copol_nrb + 2 × crosspol_nrb"

    variable_smoothed_nrb = ds.createVariable('NRB_smoothed', 'f8', ('range',))
    variable_smoothed_nrb.unit = "unitless"
    variable_smoothed_nrb.longname = "Wavelet smoothed cloud-free normalized relative backscatter profile"

    variable_depol = ds.createVariable('depol_ratio', 'f8', ('range',))
    variable_depol.unit = "unitless"
    variable_depol.longname = "Linear depolarization ratio profile"

    variable_bckscacoeff = ds.createVariable('bcksca_coeff', 'f8', ('range',))
    variable_bckscacoeff.unit = "1/km"
    variable_bckscacoeff.longname = "Total backscatter coefficient profile: β₁ + β₂"

    variable_bckscacoeff_negative_uncertainty = ds.createVariable('bcksca_coeff_negative_err', 'f8', ('range',))
    variable_bckscacoeff_negative_uncertainty.unit = "1/km"
    variable_bckscacoeff_negative_uncertainty.longname = "Inferred negative uncertainty profile for total backscatter coefficient"

    variable_bckscacoeff_positive_uncertainty = ds.createVariable('bcksca_coeff_positive_err', 'f8', ('range',))
    variable_bckscacoeff_positive_uncertainty.unit = "1/km"
    variable_bckscacoeff_positive_uncertainty.longname = "Inferred positive uncertainty profile for total backscatter coefficient"

    variable_aerosol_bckscacoeff = ds.createVariable('aerosol_bcksca_coeff', 'f8', ('range',))
    variable_aerosol_bckscacoeff.unit = "1/km"
    variable_aerosol_bckscacoeff.longname = "Aerosol backscatter coefficient profile: β₁"

    variable_aerosol_bckscacoeff_negative_uncertainty = ds.createVariable('aerosol_bcksca_coeff_negative_err', 'f8', ('range',))
    variable_aerosol_bckscacoeff_negative_uncertainty.unit = "1/km"
    variable_aerosol_bckscacoeff_negative_uncertainty.longname = "Inferred negative uncertainty profile for aerosol backscatter coefficient"

    variable_aerosol_bckscacoeff_positive_uncertainty = ds.createVariable('aerosol_bcksca_coeff_positive_err', 'f8', ('range',))
    variable_aerosol_bckscacoeff_positive_uncertainty.unit = "1/km"
    variable_aerosol_bckscacoeff_positive_uncertainty.longname = "Inferred positive uncertainty profile for aerosol backscatter coefficient"

    variable_aerosol_dry_bckscacoeff = ds.createVariable('aerosol_dry_bcksca_coeff', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff.unit = "1/km"
    variable_aerosol_dry_bckscacoeff.longname = "Dry aerosol backscatter coefficient profile: dry_β₁ = f(RH) × β₁"
    variable_aerosol_dry_bckscacoeff.description = "Use this variable to calculate aerosol, CCN, and INP vertical profiles. For example, CCN(h) = CCN(h₀) × dry_β₁(h)/dry_β₁(h₀)"

    variable_aerosol_dry_bckscacoeff_negative_uncertainty = ds.createVariable('aerosol_dry_bcksca_coeff_negative_err', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff_negative_uncertainty.unit = "1/km"
    variable_aerosol_dry_bckscacoeff_negative_uncertainty.longname = "Inferred negative uncertainty profile for dry aerosol backscatter coefficient"

    variable_aerosol_dry_bckscacoeff_positive_uncertainty = ds.createVariable('aerosol_dry_bcksca_coeff_positive_err', 'f8', ('range',))
    variable_aerosol_dry_bckscacoeff_positive_uncertainty.unit = "1/km"
    variable_aerosol_dry_bckscacoeff_positive_uncertainty.longname = "Inferred positive uncertainty profile for dry aerosol backscatter coefficient"

    variable_blind_aerosol_dry_bckscacoeff = ds.createVariable('blind_aerosol_dry_bcksca_coeff', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff.unit = "1/km"
    variable_blind_aerosol_dry_bckscacoeff.longname = "Blind range dry aerosol backscatter coefficient profile: dry_β₁ = f(RH) × β₁"
    variable_blind_aerosol_dry_bckscacoeff.description = "Use this variable to calculate aerosol, CCN, and INP vertical profiles. For example, CCN(h) = CCN(h₀) × dry_β₁(h)/dry_β₁(h₀)"

    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty = ds.createVariable('blind_aerosol_dry_bcksca_coeff_negative_err', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty.unit = "1/km"
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty.longname = "Blind range inferred negative uncertainty profile for dry aerosol backscatter coefficient"

    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty = ds.createVariable('blind_aerosol_dry_bcksca_coeff_positive_err', 'f8', ('blind_range',))
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty.unit = "1/km"
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty.longname = "Blind range inferred positive uncertainty profile for dry aerosol backscatter coefficient"

    variable_rh_profile = ds.createVariable('rh', 'f8', ('range',))
    variable_rh_profile.unit = "unitless"
    variable_rh_profile.longname = "Smoothed relative humidity profile"

    variable_aerosol_diameter = ds.createVariable('diameter', 'f8', ('size',))
    variable_aerosol_diameter.unit = "nm"
    variable_aerosol_diameter.longname = "Aerosol diameter"

    variable_aerosol_dndlogdp = ds.createVariable('dndlogdp', 'f8', ('size',))
    variable_aerosol_dndlogdp.unit = "#/cm³"
    variable_aerosol_dndlogdp.longname = "Aerosol size distribution dN/dlogDp"

    variable_ccn_ss = ds.createVariable('ccn_ss', 'f8', ('ss',))
    variable_ccn_ss.unit = "%"
    variable_ccn_ss.longname = "Supersaturations CCN concentration is evaluated at"

    variable_ccn_conc = ds.createVariable('ccn_conc', 'f8', ('ss',))
    variable_ccn_conc.unit = "#/cm³"
    variable_ccn_conc.longname = "Average CCN concentration"

    variable_ccn_conc_stdev = ds.createVariable('ccn_conc_stdev', 'f8', ('ss',))
    variable_ccn_conc_stdev.unit = "#/cm³"
    variable_ccn_conc_stdev.longname = "CCN concentration standard deviation"

    variable_ccn_conc_ste = ds.createVariable('ccn_conc_ste', 'f8', ('ss',))
    variable_ccn_conc_ste.unit = "#/cm³"
    variable_ccn_conc_ste.longname = "CCN concentration standard error"

    variable_aerosol_concentration = ds.createVariable('aerosol_concentration', 'f8')
    variable_aerosol_concentration.unit = "#/cm³"
    variable_aerosol_concentration.longname = "Aerosol concentration calculated"

    variable_ste_aerosol_concentration = ds.createVariable('aerosol_concentration_ste', 'f8')
    variable_ste_aerosol_concentration.unit = "#/cm³"
    variable_ste_aerosol_concentration.longname = "Aerosol concentration standard error"

    variable_kappa = ds.createVariable('kappa', 'f8')
    variable_kappa.unit = "unitless"
    variable_kappa.longname = "Hygroscopicity kappa calculated from CCN concentration and aerosol size distribution"

    variable_kappa_gstdev = ds.createVariable('kappa_gstdev', 'f8')
    variable_kappa_gstdev.unit = "unitless"
    variable_kappa_gstdev.longname = "Hygroscopicity kappa geometric standard deviation"

    # Assign data to variables
    variable_range[:] = inv_range
    variable_blind_range[:] = blind_inv_range
    variable_nrb[:] = nrb_profile_interp
    variable_smoothed_nrb[:] = nrb_smoothed_interp
    variable_depol[:] = depol_profile_interp
    variable_bckscacoeff[:] = bcksca_coeff_profile[0]
    variable_bckscacoeff_negative_uncertainty[:] = bcksca_coeff_profile[1]
    variable_bckscacoeff_positive_uncertainty[:] = bcksca_coeff_profile[2]
    variable_aerosol_bckscacoeff[:] = aerosol_bcksca_coeff_profile[0]
    variable_aerosol_bckscacoeff_negative_uncertainty[:] = aerosol_bcksca_coeff_profile[1]
    variable_aerosol_bckscacoeff_positive_uncertainty[:] = aerosol_bcksca_coeff_profile[2]
    variable_aerosol_dry_bckscacoeff[:] = dry_aerosol_bcksca_coeff_profile[0]
    variable_aerosol_dry_bckscacoeff_negative_uncertainty[:] = dry_aerosol_bcksca_coeff_profile[1]
    variable_aerosol_dry_bckscacoeff_positive_uncertainty[:] = dry_aerosol_bcksca_coeff_profile[2]
    variable_blind_aerosol_dry_bckscacoeff[:] = blind_profile_mae[0]
    variable_blind_aerosol_dry_bckscacoeff_negative_uncertainty[:] = blind_profile_mae[1]
    variable_blind_aerosol_dry_bckscacoeff_positive_uncertainty[:] = blind_profile_mae[2]
    variable_rh_profile[:] = smotothed_rh_profile
    variable_aerosol_diameter[:] = merged_diameter
    variable_aerosol_dndlogdp[:] = mean_dndlogdp
    variable_ccn_ss[:] = ss_CCN
    variable_ccn_conc[:] = conc_CCN
    variable_ccn_conc_stdev[:] = stdev_conc_CCN
    variable_ccn_conc_ste[:] = ste_CCN

    variable_aerosol_concentration.assignValue(aerosol_concentration)
    variable_ste_aerosol_concentration.assignValue(ste_aerosol)
    variable_kappa.assignValue(kappa)
    variable_kappa_gstdev.assignValue(kappa_gstdev)

    # INP data if available
    if np.any(temp_INP):
        dimension_temp = ds.createDimension('temp', len(temp_INP))
        variable_inp_temp = ds.createVariable('inp_temp', 'f8', ('temp',))
        variable_inp_temp.unit = "°C"
        variable_inp_temp.longname = "Temperatures at which INP concentration is evaluated"

        variable_inp_conc = ds.createVariable('inp_conc', 'f8', ('temp',))
        variable_inp_conc.unit = "#/L"
        variable_inp_conc.longname = "Average INP concentration"

        variable_inp_temp[:] = temp_INP
        variable_inp_conc[:] = conc_INP

    ds.description = f'Lidar retrieved aerosol vertical profile, aerosol size distribution, CCN spectra, and INP spectra collected at {station_name}'

    ds.close()
    


# def process_arm_mpl_data(start_time, end_time, 
#                          scale_num = 30, length_treshold = 25, value_treshold = 1.5, 
#                          edge_treshold = 1.4, contain_cloud_treshold = 1.45, cloud_pixel_number_treshold = 2,
#                          not_a_cloud_treshold = 1.4,
#                          snr_treshold = 5,
#                          blind_zone = 0.25,
#                          input_folder = ARM_mpl_folder, 
#                          fig = None, axs = None, figsize = (15,3.5), plot_range_max = 10,
#                          savefig = False, showplot = False, 
#                          output_folder = '', fig_name = None):
#     '''
#         Read the ARM MPL data from start_time to end_time, plot the curtain time series, and return the ARM MPL object.

#     '''
    
#     # Read the ARM MPL file and perform interpolation
#     ARM_file_paths = ARM_MPL.get_file_lists(input_folder, start_time, end_time)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=np.RankWarning)
#         armmpl_data = ARM_MPL(ARM_file_paths, blind_zone=blind_zone)
#     armmpl_data.interpolate_data(30, start_time = np.datetime64(start_time), end_time = np.datetime64(end_time), mov_avg_win=3)
#     first_bin_normalizing = np.nanmean(armmpl_data.interpolated_nrb_copol[:, 0])
#     normalized_copol_nrb = armmpl_data.interpolated_nrb_copol/first_bin_normalizing

#     cloud_mask, all_cloud_bottoms, all_cloud_tops = find_layers.cwt_cloud_mask(normalized_copol_nrb, armmpl_data.range, \
#             scale_num = scale_num, length_treshold = length_treshold, value_treshold = value_treshold, edge_treshold = edge_treshold, \
#             contain_cloud_treshold = contain_cloud_treshold, cloud_pixel_number_treshold = cloud_pixel_number_treshold, not_a_cloud_treshold = not_a_cloud_treshold)
    
#     cloud_free_column = np.all(cloud_mask != 1, axis=1)

#     high_snr_depol = armmpl_data.select_snr(armmpl_data.interpolated_depol_ratio,armmpl_data.interpolated_snr_copol, snr_treshold)

#     # Plotting
#     if fig is None or axs is None:
#         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
#     plotmpl.plot_mpl_2d_timeseries(armmpl_data.interpolated_datetime, armmpl_data.range, normalized_copol_nrb, fig=fig, ax=axs[0], range_max = plot_range_max, vmin=0, vmax=2, x_tick_number = 4)
#     plotmpl.plot_mpl_2d_timeseries(armmpl_data.interpolated_datetime, armmpl_data.range, high_snr_depol, fig=fig, ax=axs[1], range_max = plot_range_max, vmin=0.01, vmax=1, color_map = plotmpl.lidar_jet, colorbar_norm = 'log', x_tick_number = 4)
#     plotmpl.plot_mpl_2d_timeseries(armmpl_data.interpolated_datetime, armmpl_data.range, cloud_mask, fig=fig, ax=axs[2], range_max = plot_range_max, x_tick_number = 4, colorbar_bool = True)

#     axs[0].set_xlabel('Time (UTC)', labelpad=-10)
#     axs[1].set_xlabel('Time (UTC)', labelpad=-10)
#     axs[2].set_xlabel('Time (UTC)', labelpad=-10)

#     for ax, label in zip(axs, ['(a)', '(b)', '(c)']):
#         ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='w')
#     plt.tight_layout()

#     if savefig:
#         if fig_name is None:
#             fig_name = f'{start_time}_{end_time}_armmpl_curtain.png'
#         fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

#     # Print information:
#     print('Processing ARM MPL Data')
#     print(f'ARM MPL files: {ARM_file_paths}')
#     plt.show()

#     return armmpl_data, cloud_mask, cloud_free_column, high_snr_depol, fig, axs


# def process_tamu_mpl_data(start_time, end_time, 
#                           scale_num = 30, length_treshold = 25, value_treshold = 1.5, 
#                           edge_treshold = 1.4, contain_cloud_treshold = 1.45, cloud_pixel_number_treshold = 2,
#                           not_a_cloud_treshold = 1.4,
#                           snr_treshold = 5,
#                           blind_zone = 0.1,
#                           input_folder = tamu_mpl_folder, 
#                           fig = None, axs = None, figsize = (15,3.5), plot_range_max = 10,
#                           savefig = False, showplot = False, 
#                           output_folder = '', fig_name = None):
    
#     TAMU_file_paths  = PyMPL.get_file_list_by_start_end_datetime(input_folder, start_time, end_time)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", category=np.RankWarning)
#         mpl_object = PyMPL(TAMU_file_paths, tamu_mpl_ap_file_path, tamu_mpl_ov_file_path, tamu_mpl_dt_file_path, blind_range = blind_zone)

#     mpl_object.interpolate_data(60, start_time = np.datetime64(start_time), end_time = np.datetime64(end_time))
#     first_bin_normalizing = np.nanmean(mpl_object.interpolated_nrb_copol[:, 0])
#     normalized_copol_nrb = mpl_object.interpolated_nrb_copol/first_bin_normalizing

#     cloud_mask, all_cloud_bottoms, all_cloud_tops = find_layers.cwt_cloud_mask(normalized_copol_nrb, mpl_object.range, \
#             scale_num = scale_num, length_treshold = length_treshold, value_treshold = value_treshold, edge_treshold = edge_treshold, \
#             contain_cloud_treshold = contain_cloud_treshold, cloud_pixel_number_treshold = cloud_pixel_number_treshold, not_a_cloud_treshold = not_a_cloud_treshold)
    
#     cloud_free_column = np.all(cloud_mask != 1, axis=1)

#     high_snr_depol = mpl_object.select_snr(mpl_object.interpolated_depol_ratio, mpl_object.interpolated_snr_copol, snr_treshold)

#     # Plotting
#     if fig is None or axs is None:
#         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize)
#     plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range, normalized_copol_nrb, fig=fig, ax=axs[0], range_max = plot_range_max, vmin=0, vmax=2, x_tick_number = 4)
#     plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range, high_snr_depol, fig=fig, ax=axs[1], range_max = plot_range_max, vmin=0.01, vmax=1, color_map = plotmpl.lidar_jet, colorbar_norm = 'log', x_tick_number = 4)
#     plotmpl.plot_mpl_2d_timeseries(mpl_object.interpolated_datetime, mpl_object.range, cloud_mask, fig=fig, ax=axs[2], range_max = plot_range_max, x_tick_number = 4, colorbar_bool = True)

#     axs[0].set_xlabel('Time (UTC)', labelpad=-10)
#     axs[1].set_xlabel('Time (UTC)', labelpad=-10)
#     axs[2].set_xlabel('Time (UTC)', labelpad=-10)

#     for ax, label in zip(axs, ['(a)', '(b)', '(c)']):
#         ax.text(0.02, 0.97, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='w')
#     plt.tight_layout()

#     if savefig:
#         if fig_name is None:
#             fig_name = f'{start_time}_{end_time}_tamumpl_curtain.png'
#         fig.savefig(os.path.join(output_folder, fig_name), dpi=300)

#     # Print information:
#     print('Processing TAMU MPL Data')
#     print(f'ARM MPL files: {TAMU_file_paths}')
#     plt.show()

#     return mpl_object, cloud_mask, cloud_free_column, high_snr_depol, fig, axs