# Standard library imports
import os
import struct
import copy
import glob
from datetime import datetime

# Scientific computing
import numpy as np
import scipy

# NetCDF handling
from netCDF4 import Dataset

class PyMPL:
    """
    ------------------------------------------------------------------
    PyMPL
    ------------------------------------------------------------------
    High-level container for processing one or more MPL scans plus all ancillary calibration data (after-pulse, overlap, dead-time).

    Parameters
    ----------
    data_input : str | list[str] | dict
        Path(s) to *.mpl files **or** pre-parsed data dictionary.
    ap_input   : str | dict
        After-pulse correction file or dict.
    ov_input   : str | dict
        Overlap correction file or dict.
    dt_input   : str | dict
        Dead-time correction file or dict.
    blind_range : float, optional
        Minimum usable range [km] (bins closer than this are masked). Check the instrument manual or with the manufacture.

    Attributes created
    ------------------
    datetime : ndarray[datetime64]
        UTC timestamps for each profile.
    seconds_since_start : ndarray[int]
        Seconds since the first profile.
    range : ndarray[float]
        Bin center ranges [km] after blind-range masking.
    range_edges : ndarray[float]
        Bin edges [km] for plotting.
    bin_resolition : float
        Range resolution [km].
    laser_energy : ndarray[float]
        Laser pulse energy [mJ].
    temp_detector : ndarray[float]
        Detector temperature [°C].
    temp_telescope : ndarray[float]
        Telescope temperature [°C].
    temp_laser : ndarray[float]
        Laser temperature [°C].
    sync_pulses_seen_per_second : ndarray[int] or None
        Only present for miniMPL systems.
    number_profile : int
        Number of profiles loaded.
    raw_copol, raw_crosspol : ndarray
        Raw photon counts (counts µs⁻¹) for co- and cross-polarized channels.
    r2_corrected_copol, r2_corrected_crosspol : ndarray
        Range-squared corrected signal.
    nrb_copol, nrb_crosspol, nrb_unpol : ndarray
        Normalized relative backscatter (NRB).
    depol_ratio : ndarray
        Volume depolarization ratio.
    snr_copol, snr_crosspol : ndarray
        Signal-to-noise ratio for each channel.
    background_copol, background_crosspol : ndarray
        Background photon counts [counts/µs].
    interpolation_flag : bool
        Set to True if interpolation has been applied.
    interpolated_* : various
        Interpolated versions of most above attributes (e.g., `interpolated_nrb_copol`).

    Methods
    -------
    General
    -------
    __init__(...) :
        Initialize and process raw + calibration input.
    deepcopy() :
        Return a deep copy of the PyMPL object.
    interpolation_reset() :
        Reset all interpolated attributes to None.

    Reading Functions (Class Methods)
    ---------------------------------
    read_files(data_path, ap_path, ov_path, dt_path) :
        Read MPL and calibration data into dicts.
    read_mpl(file_path) :
        Dispatch to single or multiple MPL file reading.
    read_mpl_single_file(file_path) :
        Read one binary .mpl file.
    read_mpl_multiple_file(file_list) :
        Read multiple MPL files.
    read_afterpulse(file_path, number_bins) :
        Read afterpulse binary file.
    read_overlap(file_path) :
        Read overlap binary file.
    read_deadtime(file_path) :
        Read deadtime correction file (.bin or .csv).

    Computation and Filtering
    -------------------------
    calculate_snr(raw_data, background, background_std_dev) :
        Compute signal-to-noise ratio.
    calculate_dtcf(data) :
        Compute deadtime correction factor.
    calculate_r2_corrected(raw_data, background) :
        Apply range-squared correction.
    calculate_nrb(raw_data, background, ap_data, ap_background) :
        Compute normalized relative backscatter.
    calculate_depol_ratio() :
        Compute volume depolarization ratio.
    select_time(start_time, end_time) :
        Return new PyMPL object filtered by time.
    select_snr(data, snr_data, snr_limit) :
        Mask data below SNR threshold.

    Interpolation and Smoothing
    ---------------------------
    interpolate_data(time_resolution, start_time=None, end_time=None, ...) :
        Interpolate all key variables onto a regular time grid.
    interpolate_single_data(new_time_array, data, ...) :
        Interpolate a single variable onto new time grid.
    make_time_array(time_resolution, start_time, end_time) :
        Create regularly spaced datetime array.
    movingaverage(values, window, axis=0) :
        Apply moving average smoothing.
    _movingaverage(values, window) :
        Helper for 1D smoothing.

    I/O Functions
    -------------
    write_mpl(output_dir, filename) :
        Write current `data_dict` to a new binary .mpl file.
    output_netcdf(output_dir, filename) :
        (Placeholder) Write output to NetCDF format.

    File Selection Utilities (Static Methods)
    -----------------------------------------
    get_file_list_by_date_range(mpl_file_folder, date_range, suffix='*.mpl') :
        Return MPL files matching given date range.
    get_file_list_by_start_end_datetime(mpl_file_folder, start_datetime, end_datetime, suffix='*.mpl') :
        Return MPL files spanning the given datetime interval.
    get_date_range(start_datetime, end_datetime) :
        Return list of daily `datetime64[D]` from start to end inclusive.

    Notes
    -----
    * Range, afterpulse, overlap, and deadtime corrections are applied **eagerly** at object initialization.
    * Temporal interpolation can be applied to generate data on a regular time grid.
    * The class is intentionally heavy; use `deepcopy()` to clone a filtered subset without modifying the original.
    """

    _C = 299792458 # m.s-1 speed of light

    _record_entry = ['unit_number','version','year','month','day','hours','minutes',
        'seconds','shots_sum','trigger_frequency','energy_monitor','temp_0',
        'temp_1','temp_2','temp_3','temp_4','background_average','background_std_dev',
        'number_channels','number_bins','bin_time','range_calibraiton','number_data_bins',
        'scan_scenario_flag','number_of_background_bins','azimuth_angle','elevation_angle',
        'campass_degrees','lidar_site','wavelength','gps_latitude','gps_longitude',
        'gps_altitude','ad_data_bad_flag','data_file_version','background_average_2',
        'background_std_dev_2','mcs_mode','first_data_bin','system_type',
        'sync_pulses_seen_per_second','first_background_bin','header_size',
        'ws_used','ws_inside_temperature','ws_outside_temperature',
        'ws_inside_humidity','ws_outside_humidity','ws_dew_point',
        'ws_wind_speed','ws_wind_direction','ws_baronmetric_pressure',
        'ws_rain_rate','channel_1_data','channel_2_data']
    
    _record_format = ['uint16','uint16','uint16','uint16','uint16','uint16',
                    'uint16','uint16','uint32','int32','uint32','uint32',
                    'uint32','uint32','uint32','uint32','float32','float32',
                    'uint16','uint32','float32','float32','uint16','uint16',
                    'uint16','float32','float32','float32','char*6','uint16',
                    'float32','float32','float32','byte','byte','float32',
                    'float32','byte','uint16','byte','uint16','uint16','uint16',
                    'byte','float32','float32','float32','float32','float32',
                    'float32','int16','float32','float32','float32 array',
                    'float32 array']

    _byte_nums = {'uint8':1,'uint16':2,'int16':2,'uint32':4,'int32':4,'float32':4,'float64':8,'byte':1,'char*6':6,'float32 array':4}
    _byte_format =  {'uint8':'<B','uint16':'<H','int16':'<h','uint32':'<I','int32':'<i','float32':'<f','float64':'<d','byte':'<B','char*6':'<cccccc',
                    'float32 array':'<'+'2000f'}
    
    _ap_header = ['ap_header','ap_file_version','ap_number_channels','ap_number_bins','ap_energy','ap_background_average_copol','ap_background_average_crosspol']
    _ap_header_format = ['uint32','uint16','uint8','uint32','float64','float64','float64']


    def __init__(self, data_input, ap_input, ov_input, dt_input, blind_range = 0.1):
        
        # make sure the instance constructor can take both dictionary and file path as input
        if all([isinstance(element, dict) for element in [data_input, ap_input, ov_input, dt_input]]):
            self.data_dict, self.ap_dict, self.ov_dict, self.dt_dict = data_input, ap_input, ov_input, dt_input
        else:
            self.data_dict, self.ap_dict, self.ov_dict, self.dt_dict = self.read_files(data_input, ap_input, ov_input, dt_input)
        
        self.blind_range = blind_range

        # prepare datetime array
        year_str    = np.char.zfill(np.array(self.data_dict['year']).astype(str), 4)
        month_str   = np.char.zfill(np.array(self.data_dict['month']).astype(str), 2)
        day_str     = np.char.zfill(np.array(self.data_dict['day']).astype(str), 2)
        hours_str   = np.char.zfill(np.array(self.data_dict['hours']).astype(str), 2)
        minutes_str = np.char.zfill(np.array(self.data_dict['minutes']).astype(str), 2)
        seconds_str = np.char.zfill(np.array(self.data_dict['seconds']).astype(str), 2)
        self.datetime = np.array([x+'-'+y+'-'+z+'T'+h+':'+m+':'+s \
                                  for x,y,z,h,m,s in zip(year_str,month_str,day_str,hours_str,minutes_str,seconds_str)], dtype='datetime64')
        self.seconds_since_start = (self.datetime-self.datetime[0]).astype('timedelta64[s]').astype(int) # Seconds since first data entry

        self.laser_energy   = np.array(self.data_dict['energy_monitor'])/1000   # micro Joules (uJ)
        self.temp_detector  = np.array(self.data_dict['temp_0'])/100            # degree C
        self.temp_telescope = np.array(self.data_dict['temp_2'])/100            # degree C
        self.temp_laser     = np.array(self.data_dict['temp_3'])/100            # degree C
        if 'sync_pulses_seen_per_second' in self.data_dict: # only miniMPL has this field
            self.sync_pulses_seen_per_second = np.array(self.data_dict['sync_pulses_seen_per_second'])
        else:
            # Handle the case where 'sync_pulses_seen_per_second' does not exist
            self.sync_pulses_seen_per_second = None  # or set to a default value, e.g., np.array([]) or None

        self.number_profile = self.datetime.size

        first_data_bin_number  = self.data_dict['first_data_bin'][0]

        self.number_bins = self.data_dict['number_bins'][0]
        if np.all(np.array(self.data_dict['number_bins']) != self.number_bins):
            raise ValueError('number_bins of scans have to be the same')

        self.number_data_bins = self.data_dict['number_data_bins'][0]
        if np.all(np.array(self.data_dict['number_bins']) != self.number_bins):
            raise ValueError('number_data_bins of scans have to be the same')
        
        self.bin_time = self.data_dict['bin_time'][0] # bin width in seconds
        if np.all(np.array(self.data_dict['bin_time']) != self.bin_time):
            raise ValueError('bin_time of scans have to be the same')
        
        self.background_copol    = np.array(self.data_dict['background_average_2'])  # count per micro seconds (count us-1)
        self.background_crosspol = np.array(self.data_dict['background_average'])    # count per micro seconds (count us-1)
        
        #self.no_blind_range      = 0.5*self.bin_time*self._C*(np.arange(self.number_bins) + 1)*1e-3 #km
        # a previous version uses +0.5 following Peter Kuma. However, using +1 generates the same r2corrected result as the sigmaMPL software
        self.no_blind_range = 0.5*self.bin_time*self._C*(np.arange(self.number_data_bins) + 0.5)*1e-3 #km
        above_blind_range   = self.no_blind_range > self.blind_range

        self.range          = self.no_blind_range[above_blind_range] # do not include range below blind range
        self.bin_resolition = self.range[1]-self.range[0]
        self.range_edges    = np.append(self.range-self.bin_resolition/2, self.range[-1]+self.bin_resolition/2)

        self.raw_copol      = np.array(self.data_dict['channel_2_data'])[:, first_data_bin_number:][:,above_blind_range] # count per micro seconds (count us-1)
        self.raw_crosspol   = np.array(self.data_dict['channel_1_data'])[:, first_data_bin_number:][:,above_blind_range] # count per micro seconds (count us-1)

        self.snr_copol      = self.calculate_snr(self.raw_copol, self.background_copol, np.array(self.data_dict['background_std_dev_2']))     # unitless
        self.snr_crosspol   = self.calculate_snr(self.raw_crosspol, self.background_crosspol, np.array(self.data_dict['background_std_dev'])) # unitless
        ## Calculated product ##
        # Calculate Range Correction Product
        self.r2_corrected_copol    = self.calculate_r2_corrected(self.raw_copol, self.background_copol)        # counts km2 per micro seconds (counts km2 us-1)
        self.r2_corrected_crosspol = self.calculate_r2_corrected(self.raw_crosspol, self.background_crosspol)  # counts km2 per micro seconds (counts km2 us-1)

        # Calculate Normalized Relative Backscatter
        self.nrb_copol    = self.calculate_nrb(self.raw_copol,    self.background_copol    \
                                               ,self.ap_dict['ap_copol'],self.ap_dict['ap_background_average_copol'])       # counts km2 per micro seconds per micro Joules (counts km2 us-1 uJ-1)
        self.nrb_crosspol = self.calculate_nrb(self.raw_crosspol, self.background_crosspol \
                                               ,self.ap_dict['ap_crosspol'],self.ap_dict['ap_background_average_crosspol']) # counts km2 per micro seconds per micro Joules (counts km2 us-1 uJ-1)
        self.nrb_unpol    = self.nrb_copol + 2 * self.nrb_crosspol

        self.depol_ratio  = self.calculate_depol_ratio()    # unitless

        ## Interpolation product ##
        self.interpolation_reset()

    def deepcopy(self):
        return copy.deepcopy(self)
        
    @classmethod
    def read_files(cls, data_path, ap_path, ov_path, dt_path):
        '''
        data_path: Path to mpl data or mpl data folder. Could also be a list of mpl data path.
        ap_path: Path to afterpulse correction file
        ov_path: Path to overlap correction file
        dt_path: Path to deadtime correction file
        '''
        # Read MPL data
        data_dict, number_bins = cls.read_mpl(data_path)            # Read MPL data
        ap_dict   = cls.read_afterpulse(ap_path, number_bins)       # Read Afterpulse data
        ov_dict   = cls.read_overlap(ov_path)                       # Read Overlap data
        dt_dict   = cls.read_deadtime(dt_path)                      # Read Deadtime data
        return data_dict, ap_dict, ov_dict, dt_dict

    @classmethod
    def read_mpl_single_file(cls, file_path):
        fin = open(file_path, "rb")
        data_dict = {}
        for i in range(len(cls._record_entry)):
            data_dict[cls._record_entry[i]] = []
        loop_bool = True
        number_bins = None
        while loop_bool:
            for i in range(len(cls._record_entry)):
                entry_type = cls._record_entry[i]
                entry_format = cls._record_format[i]
                if entry_type == 'channel_1_data' or entry_type == 'channel_2_data': 
                    byte_numbers_to_read = cls._byte_nums[entry_format] * number_bins
                else:
                    byte_numbers_to_read = cls._byte_nums[entry_format]
                binary_entry =  fin.read(byte_numbers_to_read)
                if binary_entry == b'': # check if reached the end of the file, if so then stop the loop and do not process
                    loop_bool = False
                else:
                    if entry_format in ['uint16','uint32','int16','int32','float32']:
                        unpacked_entry = struct.unpack(cls._byte_format[entry_format], binary_entry)
                        data_dict[entry_type].append(unpacked_entry[0])
                        if entry_type == 'number_bins':
                            number_bins = unpacked_entry[0]
                    elif entry_format == 'byte':
                        data_dict[entry_type].append(bin(int.from_bytes(binary_entry, byteorder='little'))[2:].zfill(8))
                    elif entry_format == 'char*6':
                        data_dict[entry_type].append(binary_entry.decode('latin-1'))
                    elif entry_format == 'float32 array':
                        #unpacked_entry = struct.unpack(cls._byte_format[entry_format], binary_entry)
                        unpacked_entry = struct.unpack('<'+str(number_bins)+'f', binary_entry)
                        data_dict[entry_type].append(list(unpacked_entry))
        fin.close()
        return data_dict, number_bins

    @classmethod
    def read_mpl_multiple_file(cls, file_path_list):
        data_dict = {}
        for i in range(len(cls._record_entry)):
            data_dict[cls._record_entry[i]] = []
        for a_path in file_path_list:
            data_dict_to_merge, number_bins = cls.read_mpl_single_file(a_path)
            for i in range(len(cls._record_entry)):
                data_dict[cls._record_entry[i]].extend(data_dict_to_merge[cls._record_entry[i]])
        return data_dict, number_bins

    @classmethod
    def read_mpl(cls, file_path):
        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                # read all mpl files contained in a folder
                filelist = glob.glob(os.path.join(file_path, '*.mpl'))
                filelist.sort(key=os.path.getmtime)
                return cls.read_mpl_multiple_file(filelist)
            elif os.path.isfile(file_path):
                # read a single mpl file
                return cls.read_mpl_single_file(file_path)
        elif isinstance(file_path, list):
            # read all mpl files from paths contianed in a list
            return cls.read_mpl_multiple_file(file_path)
    
    @classmethod
    def read_afterpulse(cls, file_path, number_bins):
        fin = open(file_path, "rb")
        ap_dict = {}
        for i in range(len(cls._ap_header)):
            entry_type = cls._ap_header[i]
            entry_format = cls._ap_header_format[i]
            byte_numbers_to_read = cls._byte_nums[entry_format]
            binary_entry =  fin.read(byte_numbers_to_read)
            ap_dict[entry_type] = struct.unpack(cls._byte_format[entry_format], binary_entry)[0]
        ap_number_bins = int(ap_dict['ap_number_bins'])
        for x in ['ap_range', 'ap_copol', 'ap_crosspol']:
            buf = fin.read(8*ap_number_bins)
            a = struct.unpack_from('<' + 'd'*ap_number_bins, buf)
            ap_dict[x] = np.array(a)
        fin.close()
        return ap_dict
        
    @classmethod
    def read_overlap(cls, file_path):
        fin = open(file_path, "rb")
        buf = fin.read()
        n = len(buf)//16
        fin.seek(0)
        ov_dict = {'ol_number_bins': n}
        for x in ['ol_range', 'ol_overlap']:
            binary_entry = fin.read(8*n)
            if len(binary_entry) < 8*n:
                raise IOError('Incomplete %s data' % x)
            ov_dict[x] = np.array(struct.unpack_from('<' + 'd'*n, binary_entry))
        fin.close()
        return ov_dict
    
    @classmethod
    def read_deadtime(cls, file_path):

        file_extension = os.path.splitext(file_path)[-1]

        if file_extension == '.bin':
            fin = open(file_path, "rb")
            buf = fin.read()
            n = len(buf)//4
            dt_dict = {'dt_number_coeff': n}
            dt_dict['dt_coeff'] = np.array(struct.unpack_from('<' + 'f'*n, buf), np.float32)
            dt_dict['dt_coeff_degree'] = np.arange(n - 1, -1, -1)
        elif file_extension == '.csv':
            coefficient_data = np.loadtxt(file_path, delimiter=",")
            n = len(coefficient_data)
            dt_dict = {'dt_number_coeff': n}
            dt_dict['dt_coeff'] = coefficient_data
            dt_dict['dt_coeff_degree'] = np.arange(n - 1, -1, -1)
        else:
            raise IOError('deadtime correction file in .bin or .csv format')
        return dt_dict

    def calculate_snr(self, raw_data, background, background_std_dev):
        background_stacked = np.transpose(np.vstack([background] * raw_data.shape[1]))
        background_std_dev_stacked = np.transpose(np.vstack([background_std_dev] * raw_data.shape[1]))

        return (raw_data - background_stacked)/background_std_dev_stacked

    def calculate_dtcf(self, data): # calculate deadtime correction factor
        dt_coeff = self.dt_dict['dt_coeff']
        dt_coeff_degree = self.dt_dict['dt_coeff_degree']
        dt_number_coeff = self.dt_dict['dt_number_coeff']
        return np.sum([(data*1e3)**(dt_coeff_degree[i])*dt_coeff[i] for i in range(dt_number_coeff)], axis=0)

    def calculate_r2_corrected(self, raw_data, background): # the result is exactly the same as the manufacture mpl software result
        background_stacked = np.transpose(np.vstack([background]*raw_data.shape[1]))
        return (raw_data - background_stacked)*np.square(self.range)


    def calculate_nrb(self, raw_data, background, ap_data, ap_background):
        ap_range = self.ap_dict['ap_range']
        ol_range = self.ov_dict['ol_range']
        ov_data  = self.ov_dict['ol_overlap']
        ap_data_interped = np.interp(self.range, ap_range, ap_data)
        ov_data_interped = np.interp(self.range, ol_range, ov_data)
        
        ap_data_stacked  = np.vstack([ap_data_interped] * raw_data.shape[0])
        ov_data_stacked  = np.vstack([ov_data_interped] * raw_data.shape[0])
        energy_stacked   = np.transpose(np.vstack([self.laser_energy] * raw_data.shape[1]))
        range_stacked    = np.vstack([self.range] * raw_data.shape[0])

        dt_corrected_raw = raw_data * self.calculate_dtcf(raw_data)
        background_correction = np.transpose(np.vstack([background * self.calculate_dtcf(background)] * raw_data.shape[1]))
        afterpulse_correction = (
            ap_data_stacked * self.calculate_dtcf(ap_data_stacked) - ap_background  * self.calculate_dtcf(ap_background)
        ) * energy_stacked  / self.ap_dict['ap_energy']

        nrb = (dt_corrected_raw - background_correction - afterpulse_correction) * np.square(range_stacked) / (ov_data_stacked*energy_stacked)
        return nrb

    def calculate_depol_ratio(self):
        cross_over_co = self.nrb_crosspol/self.nrb_copol
        return cross_over_co/(cross_over_co+1)

    def select_time(self, start_time, end_time): # return a new PyMPL object with data within the start_time and end_time
        start_time = np.datetime64(start_time)  # include start
        end_time   = np.datetime64(end_time)    # include end
        selected_indicies = np.where(np.logical_and(self.datetime>=start_time, self.datetime<=end_time))[0]

        selected_data_dict = {}
        for key, value in self.data_dict.items():
            selected_data_dict[key] = [value[i] for i in selected_indicies]
        return PyMPL(selected_data_dict, self.ap_dict, self.ov_dict, self.dt_dict)

    @classmethod
    def select_snr(cls, data, snr_data, snr_limit): # return data with a minimum snr
        high_snr_data = data.copy()
        high_snr_data[snr_data<snr_limit] = np.nan
        return high_snr_data
    
    def interpolate_single_data(self, new_time_array, data, time_resolution=None, gap_seconds = None):
        '''
        time_resolution: time resolution in seconds
        gap_seconds: determine when the gap in timestamps is too large to be interpolated
        '''

        if time_resolution is None:
            time_resolution = np.nanmean(new_time_array[1:]-new_time_array[:-1])

        data = np.array(data) # make sure its an np array
        if gap_seconds is None:
            gap_seconds = time_resolution * 2

        new_seconds_passed = (new_time_array-new_time_array[0]).astype('timedelta64[s]').astype(int)
        original_seconds_passed  = (self.datetime-new_time_array[0]).astype('timedelta64[s]').astype(int)

        # this works because it finds the actural datetime of the gap so it does not matter where it started
        time_difference    = self.seconds_since_start[1:] - self.seconds_since_start[:-1] 
        gap_indicies       = np.where(time_difference>gap_seconds)[0]
        gap_starts_arr     = self.datetime[gap_indicies]
        gap_ends_arr       = self.datetime[gap_indicies+1]

        linear_interpolator = scipy.interpolate.interp1d(original_seconds_passed, data, kind='linear', axis= 0, bounds_error = False, fill_value = np.nan)
        interpolated_data   = linear_interpolator(new_seconds_passed) # is a numpy array

        for i in range(len(gap_indicies)):
            gap_starts = gap_starts_arr[i]
            gap_ends   = gap_ends_arr[i]
            new_gap_indicies  = np.where(np.logical_and(new_time_array>gap_starts, new_time_array<gap_ends))[0]
            interpolated_data[new_gap_indicies] = np.nan

        return interpolated_data
    
    @classmethod
    def make_time_array(cls, time_resolution, start_time, end_time):
        start_time = np.datetime64(start_time)
        end_time   = np.datetime64(end_time) + np.timedelta64(time_resolution,'s') # so the end time is also included

        return np.arange(start_time, end_time, np.timedelta64(time_resolution, 's'))
    
    def interpolate_data(self, time_resolution, start_time = None, end_time = None, gap_seconds = None, mov_avg_win = None):
        '''
        time_reoslution in seconds
        '''

        if start_time is None:
            start_time = self.datetime[0]

        if end_time is None:
            end_time = self.datetime[-1] 

        if gap_seconds is None:
            gap_seconds = time_resolution * 2
        
        # return a new PyMPL object with interpolated data within the start_time and end_time
        
        # Generate a new time array
        new_time_array = self.make_time_array(time_resolution, start_time, end_time)
        seconds_since_start = (new_time_array - new_time_array[0]).astype('timedelta64[s]').astype(int)

        # List of attributes to interpolate
        attributes = ['laser_energy', 'temp_detector', 'temp_telescope', 'temp_laser',
                    'raw_copol', 'raw_crosspol', 'r2_corrected_copol', 'r2_corrected_crosspol',
                    'nrb_copol', 'nrb_crosspol', 'nrb_unpol', 'depol_ratio', 'snr_copol', 'snr_crosspol'
        ]

        if 'sync_pulses_seen_per_second' in self.data_dict:
            attributes.append('sync_pulses_seen_per_second')

        # Enable interpolation flag
        self.interpolation_flag = True
        self.interpolated_datetime = new_time_array
        self.interpolated_seconds_since_start = seconds_since_start

        # Function to apply moving average and interpolation
        def process_attribute(attr):
            data = getattr(self, attr)
            if mov_avg_win:
                data = self.movingaverage(data, mov_avg_win, axis=1)
            return self.interpolate_single_data(new_time_array, data, time_resolution=time_resolution, gap_seconds=gap_seconds)

        # Apply interpolation to each attribute
        for attr in attributes:
            setattr(self, f'interpolated_{attr}', process_attribute(attr))

    def interpolation_reset(self):
    # List of interpolated attributes to reset
        attributes = [
            'datetime', 'seconds_since_start', 'laser_energy', 'temp_detector', 
            'temp_telescope', 'temp_laser', 'raw_copol', 'raw_crosspol', 
            'r2_corrected_copol', 'r2_corrected_crosspol', 'nrb_copol', 
            'nrb_crosspol', 'nrb_unpol', 'depol_ratio', 'snr_copol', 'snr_crosspol'
        ]
        if 'sync_pulses_seen_per_second' in self.data_dict:
            attributes.append('sync_pulses_seen_per_second')

        # Reset the interpolation flag
        self.interpolation_flag = False

        # Reset all attributes to None
        for attr in attributes:
            setattr(self, f'interpolated_{attr}', None)
    
    def write_mpl(self, output_dir, filename):
        """
        output the current data dict as a new binary mpl file.
        Useful when need to need to modify the .mpl binary file by changing self.data_dict
        """
        output_path = os.path.join(output_dir, filename+'.mpl')
        new_dict = self.data_dict.copy()
        
        new_file = open(output_path, 'wb')
        for i in range(self.number_profile):
            for j in range(len(self._record_entry)):
                entry_type = self._record_entry[j]
                entry_format = self._record_format[j]
                data_to_write = new_dict[entry_type][i]
                if entry_format in ['uint16','uint32','int16','int32','float32']:
                    binary_data = struct.pack(self._byte_format[entry_format], data_to_write)
                elif entry_format == 'byte':
                    integer_value = int(data_to_write, 2)
                    binary_data = struct.pack(self._byte_format[entry_format], integer_value)
                elif entry_format == 'char*6':
                    binary_data = data_to_write.encode('latin-1')
                elif entry_format == 'float32 array':
                    binary_data = struct.pack(self._byte_format[entry_format], *data_to_write)
                    
                new_file.write(binary_data)

    def output_netcdf(self, output_dir, filename):
        ncfile = Dataset(os.path.join(output_dir, filename+'.nc'),mode='w',format='NETCDF4')
        pass

    @classmethod
    def _movingaverage(cls, values, window):
        # helper function moving average with 1d array
        window_half = window // 2
        nan_paddings = np.full(window_half, np.nan)

        front_paddings = np.full(window_half, np.nanmean(values[:window_half]))
        end_paddings   = np.full(window_half, np.nanmean(values[-window_half:]))

        # Create a 2D view of the padded array with a rolling window
        rolling_view = np.lib.stride_tricks.sliding_window_view(values, window)
        
        # Calculate the mean along the rolling window axis while ignoring NaNs
        smoothed_data = np.nanmean(rolling_view, axis=1)
        smoothed_data = np.hstack([front_paddings, smoothed_data, end_paddings])
        return smoothed_data

    @classmethod
    def movingaverage(cls, values, window, axis = 0):
        # values could either be an 1d array or a 2d array.
        values = np.array(values)
        if values.ndim == 1:
            return cls._movingaverage(values, window)
        elif values.ndim == 2:
            result = []
            for i in range(values.shape[axis]):
                if axis == 0:
                    result.append(cls._movingaverage(values[i], window))
                elif axis == 1:
                    result.append(cls._movingaverage(values[:,i], window))
                else:
                    raise ValueError('axis has to be 0 or 1')
            result = np.array(result)
            if axis == 1:
                result = np.transpose(result)
            return result
        else:
            raise ValueError('values has to be an 1D array or a 2D array')
        

    @staticmethod
    def get_file_list_by_date_range(mpl_file_folder, date_range, suffix='*.mpl'):
        filelist = glob.glob(os.path.join(mpl_file_folder, suffix))
        filelist.sort(key=os.path.getmtime)
        substring_list = [date.item().strftime("%Y%m%d") for date in date_range]
        selected_file_list = [s for s in filelist if any(substring in s for substring in substring_list)]

        return selected_file_list
    
    @staticmethod
    def get_file_list_by_start_end_datetime(mpl_file_folder, start_datetime, end_datetime, suffix='*.mpl'):
        '''
            for tamu mpl files
        '''
        filelist = glob.glob(os.path.join(mpl_file_folder, suffix))
        filelist.sort(key=os.path.getmtime)

        start_str = (start_datetime.astype(datetime)).strftime("%Y%m%d%H%M")
        end_str = (end_datetime.astype(datetime)).strftime("%Y%m%d%H%M")

        # Initialize indices for the files immediately before and after the date range
        index_before_start = -1
        index_after_end = len(filelist)

        # Iterate through the file list to find the relevant indices
        for i, file in enumerate(filelist):
            file_base = os.path.basename(file)[:-4]
            if file_base < start_str and (index_before_start == -1 or file_base > os.path.basename(filelist[index_before_start])[:-4]):
                index_before_start = i
            if file_base > end_str and (index_after_end == len(filelist) or file_base < os.path.basename(filelist[index_after_end])[:-4]):
                index_after_end = i
                break  # No need to continue once the file after end is found

        # Filter files based on datetime range
        selected_file_list = [s for s in filelist if os.path.basename(s)[:-4] >= start_str and os.path.basename(s)[:-4] <= end_str]

        # Add the file immediately before the start time if it exists
        if index_before_start != -1:
            selected_file_list.insert(0, filelist[index_before_start])

        # Add the file immediately after the end time if it exists
        if index_after_end < len(filelist):
            selected_file_list.append(filelist[index_after_end])

        return  sorted(selected_file_list)
        #return  selected_file_list
    
    @staticmethod
    def get_date_range(start_datetime, end_datetime):
        return np.arange(start_datetime, end_datetime + np.timedelta64(1, 'D') , np.timedelta64(1, 'D'), dtype='datetime64[D]')
    