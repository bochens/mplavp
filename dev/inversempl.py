import numpy as np
import scipy
import matplotlib.pyplot as plt
import pywt

#Extinction to backscatter ratio for molecular scatterers
molecular_lidar_ratio = 8*np.pi/3 

# Read U.S. Standard Atmosphere
_atmosphere_data = np.loadtxt('/Users/bochen/TAMU/mpl-py/us_standard.csv', skiprows=1, delimiter=',') # Read US Standard Atmosphere data
_height   = _atmosphere_data[:,0]
_pressure = _atmosphere_data[:,1] * 100 #pa (file uses hpa)
_temp_K   = _atmosphere_data[:,2]

US_standard_pressure    = scipy.interpolate.interp1d(_height, _pressure, bounds_error=False, fill_value=np.nan) # Interpolate US standard Atmosphere data
US_standard_temperature = scipy.interpolate.interp1d(_height, _temp_K, bounds_error=False, fill_value=np.nan)

def calculate_backscatter_coefficient(LAM, pressure, temperature):
    '''
    Using equations from Gray G. Gimmestad and David W. Roberts Page 40
    input: wavelength (um), pressure (pa), temperature (K) 
    return: volume bsckscatter coefficient (km-1 sr-1)
    '''
    stp_backscatter_coeff = 1.39 * (0.55/LAM)**4 * 1E-6 # m-1sr-1
    return stp_backscatter_coeff * (pressure * 288.15) / (101325 * temperature) * 1000 #km-1sr-1

def rayleigh_bcksca_coeff(LAM, height, pressure_interpolater = US_standard_pressure, temperature_interpolater = US_standard_temperature):
    '''
    molecular backscatter of atmosphere at height. 
    Default is US standard atmosphere defined in John H. Seinfeld, Atmospheric Chemistry and Physics
    LAM in um, height in km
    '''
    return calculate_backscatter_coefficient(LAM, pressure_interpolater(height), temperature_interpolater(height))


def Fernald_inversion_inwards(nrb, mpl_range, beta2, S1, S2 = molecular_lidar_ratio, calibration_beta1 = 0, calibration_range = 20, LAM = 0.532):
    '''
    Inwards 2-Components Fixed Lidar-ratio Fernald Method Inversion for Backscatter Coefficient Profile
    This function is adapted from Equation (6) of Frederick G. Fernald, Analysis of atmospheric lidar observations: some comments. 1984. Applied Optics
    input:
        nrb: 1d or 2d attenuated backscatter data. dimension is [timestamp (optional), range]; unit is (counts km2 us-1 uJ-1)
        lidar_range: height in km, equal intervals.
        beta2: rayleigh backscatter coefficient profile. Dimension can be 1d or same to nrb dimension
        S1: Lidar ratio (extinction to backscatter ratio) for particulate scatterers
        S2: Lidar ratio for molecular scatterers
        calibration_beta1: Aerosol backscatter coefficient calibration value at calibration range. A number or a 1D array unit in (km-1)
        calibration_range: Calibration range
        LAM: lidar wavelength in um. Default to 532 nm for micropulse lidar

    return:
        beta1: retrieved backscatter coefficient (km-1 sr-1). dimension is [timestamp (optional), range]
        inv_range: range used for the inversion
    '''

    nrb = np.transpose(np.array(nrb)) # transpose so the dimension is now [range, timestamp] since timestamp is optional
    beta2 = np.transpose(np.array(beta2))
    selected_range = mpl_range[mpl_range<=calibration_range]
    selected_nrb   = nrb[mpl_range<=calibration_range]  #counts * km2 / micro s micro j 
    selected_beta2 = beta2[mpl_range<=calibration_range]

    delta_r = selected_range[1:]-selected_range[:-1] #km
    total_backscatter = list(range(selected_range.size))
    calibration_beta2 = selected_beta2[-1]
    calibration_backscatter = calibration_beta1+calibration_beta2

    if (nrb.ndim - calibration_backscatter.ndim == 1):
        beta2_output = selected_beta2
        total_backscatter[-1] = calibration_backscatter # calibration data
    elif nrb.ndim - calibration_backscatter.ndim == 2:
        beta2_output = np.array([selected_beta2]*nrb.shape[1])
        total_backscatter[-1] = np.array([calibration_backscatter]*nrb.shape[1])
    else:
        raise ValueError('nrb can have 1 or 2 more dimensions than the boundary condition.\
                        beta2 can be a 2D array or an 1D array. calibration_beta1 can be a 1D array or a number')
    
    for I in reversed(range(selected_range.size)): 
        if I>0:
            A_term = (S1-S2) * (selected_beta2[I-1]+selected_beta2[I]) * delta_r[I-1]
            numera = selected_nrb[I-1] * np.exp(A_term)
            C_term = selected_nrb[I]/total_backscatter[I]
            denomi = C_term + S1*(selected_nrb[I]+selected_nrb[I-1] * np.exp(A_term)) * delta_r[I-1]
            total_backscatter[I-1] = numera / denomi  # Each loop is calculating for i-1
        else:
            pass

    return np.transpose(np.array(total_backscatter)), beta2_output, selected_range



def Sasano_inversion_inwards(nrb, mpl_range, beta2, S1_function, S2 = molecular_lidar_ratio, calibration_beta1 = 0, calibration_range = 20, LAM = 0.532):

    '''
    Variable aerosol lidar ratio. Frin Sasabi et al. Error caused by using a constant extinction/backscatteirng raito in the lidar solution
    '''

    nrb = np.transpose(np.array(nrb)) # transpose so the dimension is now [range, timestamp] since timestamp is optional
    beta2 = np.transpose(np.array(beta2))
    selected_range = mpl_range[mpl_range<=calibration_range]
    selected_nrb   = nrb[mpl_range<=calibration_range]  #counts * km2 / micro s micro j 
    selected_beta2 = beta2[mpl_range<=calibration_range]

    delta_r = np.mean(selected_range[1:]-selected_range[:-1]) #km
    total_backscatter = list(range(selected_range.size))
    calibration_beta2 = selected_beta2[-1]
    calibration_backscatter = calibration_beta1+calibration_beta2

    if (nrb.ndim - calibration_backscatter.ndim == 1):
        beta2_output = selected_beta2
        total_backscatter[-1] = calibration_backscatter # calibration data
    elif nrb.ndim - calibration_backscatter.ndim == 2:
        beta2_output = np.array([selected_beta2]*nrb.shape[1])
        total_backscatter[-1] = np.array([calibration_backscatter]*nrb.shape[1])
    else:
        raise ValueError('nrb can have 1 or 2 more dimensions than the boundary condition.\
                        beta2 can be a 2D array or an 1D array. calibration_beta1 can be a 1D array or a number')
    
    S1 = S1_function(selected_range)
    
    for i in reversed(range(selected_range.size)): 
        if i>0:
            A_term = ( (S1[i-1]-S2) * selected_beta2[i-1] + (S1[i]-S2) * selected_beta2[i] ) * delta_r

            numera = selected_nrb[i-1] * np.exp(A_term)
            C_term = selected_nrb[i]/total_backscatter[i]
            denomi = C_term + (S1[i]*selected_nrb[i] + S1[i-1]*selected_nrb[i-1]*np.exp(A_term)) * delta_r
            total_backscatter[i-1] = numera / denomi  # Each loop is calculating for i-1
        else:
            pass
    


def Fernald_inversion_outwards(nrb, mpl_range, beta2, S1, S2 = molecular_lidar_ratio, calibration_beta1 = 0, calibration_range = None, LAM = 0.532):
    '''
    Outwards 2-Components Fixed Lidar-ratio Fernald Method Inversion for Backscatter Coefficient Profile
    input:
        nrb: 1d or 2d attenuated backscatter data. dimension is [timestamp (optional), range]; unit is (counts km2 us-1 uJ-1)
        lidar_range: height in km, equal intervals.
        beta2: rayleigh backscatter coefficient profile. Dimension can be 1d or same to nrb dimension
        S1: Lidar ratio (extinction to backscatter ratio) for particulate scatterers
        S2: Lidar ratio for molecular scatterers
        calibration_beta1: Aerosol backscatter coefficient calibration value at calibration range. A number or a 1D array
        calibration_range: Calibration range
        LAM: lidar wavelength in um. Default to 532 nm for micropulse lidar

    return:
        beta1: retrieved backscatter coefficient (km-1 sr-1). dimension is [timestamp (optional), range]
        inv_range: range used for the inversion
    '''

    # Test if S1 (aerosol lidar ratio) can be a 1 dimensional array when the nrb is 2 dimensional

    if calibration_range is None:
        calibration_range = mpl_range[0]

    nrb = np.transpose(np.array(nrb)) # transpose so the dimension is now [range, timestamp] since timestamp is optional
    beta2 = np.transpose(np.array(beta2)) # same as last comment

    selected_range = mpl_range[mpl_range>=calibration_range]
    selected_nrb   = nrb[mpl_range>=calibration_range]  #counts * km2 / micro s micro j 
    selected_beta2 = beta2[mpl_range>=calibration_range]

    delta_r = np.mean(selected_range[1:]-selected_range[:-1]) #km
    total_backscatter = list(range(selected_range.size))
    calibration_beta2 = selected_beta2[0]
    calibration_backscatter = calibration_beta1+calibration_beta2

    if (nrb.ndim - calibration_backscatter.ndim == 1):
        beta2_output = selected_beta2
        total_backscatter[0] = calibration_backscatter # calibration data
    elif nrb.ndim - calibration_backscatter.ndim == 2:
        beta2_output = np.array([selected_beta2]*nrb.shape[1])
        total_backscatter[0] = np.array([calibration_backscatter]*nrb.shape[1])
    else:
        raise ValueError('nrb can have 1 or 2 more dimensions than the boundary condition.\
                        beta2 can be a 2D array or an 1D array. calibration_beta1 can be a 1D array or a number')
    
    for i in range(selected_range.size-1):
        A_term = (S1-S2) * (selected_beta2[i]+selected_beta2[i+1]) * delta_r
        numera = selected_nrb[i+1] * np.exp(-A_term)
        C_term = selected_nrb[i]/total_backscatter[i]
        denomi = C_term - S1*(selected_nrb[i]+selected_nrb[i+1] * np.exp(-A_term)) * delta_r
        total_backscatter[i+1] = numera / denomi  # Each loop is calculating for i+1

    return np.transpose(np.array(total_backscatter)), beta2_output, selected_range

def Klett_inversion_inwards(nrb, mpl_range, S1, calibration_beta = 0, calibration_range = 20):
    '''
    Inwards 1-Component Fixed Lidar-ratio Klett Method Inversion for Bakcscatter Coefficient Profile
    This function is adapted from Equation (9) of Frederick G. Fernald, Analysis of atmospheric lidar observations: some comments. 1984. Applied Optics
    input:
        calibration_beta can be a number or an 1D array with time dimension
    '''

    nrb = np.transpose(np.array(nrb))
    selected_range = mpl_range[mpl_range<=calibration_range]
    selected_nrb   = nrb[mpl_range<=calibration_range]

    delta_r = np.mean(selected_range[1:]-selected_range[:-1]) #km
    total_backscatter = list(range(selected_range.size))
    
    if nrb.ndim == 1:
        if isinstance(calibration_beta, (int, float)): # nrb is 1D array and calibration_beta is a number
            total_backscatter[-1] = calibration_beta
        else:
            raise ValueError('calibration_beta has to be a number when nrb is an 1D array')
    elif nrb.ndim == 2:
        if isinstance(calibration_beta, (int, float)):
            total_backscatter[-1] = np.array([calibration_beta]*nrb.shape[1]) # nrb is 1D array and calibration_beta is a 1D array
        elif isinstance(np.array(calibration_beta), np.ndarray) and np.array(calibration_beta).ndim == 1:
            total_backscatter[-1] = np.array(calibration_beta)
        else:
            raise ValueError('calibration_beta has to be a number or an 1D array when nrb is a 2D array')
    else:
        raise ValueError('nrb has to be an 1D array or a 2D array')

    for i in reversed(range(selected_range.size)):
        if i>0:
            numera = selected_nrb[i-1]
            C_term = selected_nrb[i]/total_backscatter[i]
            denomi = C_term + (selected_nrb[i] + selected_nrb[i-1])*delta_r
            total_backscatter[i-1] = numera / denomi  # Each loop is calculating for i-1
        else:
            pass

    return np.transpose(np.array(total_backscatter)), selected_range


def iterative_aod_Fernald_inversion_inwards(nrb, mpl_range, beta2, S1, aod, calibration_beta1 = 0, calibration_range = 20, LAM = 0.532, criteria = 0.005):

    new_aerosol_lidar_ratio = S1
    epsilon = criteria
    iterat_num = 0
    while epsilon >= criteria:
        print(f'lidar ratio is {new_aerosol_lidar_ratio}')
        bcksca_coeff_1d_inward, beta2_output_1d_inward, inv_range_1d_inward \
                = Fernald_inversion_inwards(nrb, mpl_range, beta2, \
                S1 = new_aerosol_lidar_ratio, calibration_beta1 = calibration_beta1, calibration_range=calibration_range)
        
        bcksca_coeff_aerosol_inward = bcksca_coeff_1d_inward-beta2_output_1d_inward
        baksca_coeff_integral = scipy.integrate.simpson(y = bcksca_coeff_aerosol_inward, x = inv_range_1d_inward)
        baksca_coeff_integral = baksca_coeff_integral + bcksca_coeff_aerosol_inward[0] * (mpl_range[0]-0)
        
        last_lidar_ratio = new_aerosol_lidar_ratio
        new_aerosol_lidar_ratio = aod / baksca_coeff_integral

        epsilon = np.absolute(new_aerosol_lidar_ratio-last_lidar_ratio)
        
        iterat_num = iterat_num+1
    
    return bcksca_coeff_1d_inward, beta2_output_1d_inward, inv_range_1d_inward, last_lidar_ratio

# Adapted from https://notebook.community/CSchoel/learn-wavelets/wavelet-denoising
# Plot the whole range of coefficients for a full decomposition with the DWT
def plot_dwt(details, approx, xlim=(-300,300), **line_kwargs): 
    for i in range(len(details)):
        plt.subplot(len(details)+1,1,i+1)
        d = details[len(details)-1-i]
        half = len(d)//2
        xvals = np.arange(-half,-half+len(d))* 2**i
        plt.plot(xvals, d, **line_kwargs)
        #plt.xlim(xlim)
        plt.title("detail[{}]".format(i))
    plt.subplot(len(details)+1,1,len(details)+1)
    plt.title("approx")
    plt.plot(xvals, approx, **line_kwargs)
    #plt.xlim(xlim)

# Adapted from https://notebook.community/CSchoel/learn-wavelets/wavelet-denoising
# Based on Incorporating Information on Neighbouring Coefficients into Wavelet Estimation
def neigh_block(details, n, sigma):
    res = []
    L0 = int(np.log2(n) // 2)
    L1 = max(1, L0 // 2)
    L = L0 + 2 * L1
    def nb_beta(sigma, L, detail):
        S2 = np.sum(detail ** 2)
        lmbd = 4.50524 # solution of lmbd - log(lmbd) = 3
        beta = (1 - lmbd * L * sigma**2 / S2)
        return max(0, beta)
    for d in details:
        d2 = d.copy()
        for start_b in range(0, len(d2), L0):
            end_b = min(len(d2), start_b + L0)
            start_B = start_b - L1
            end_B = start_B + L
            if start_B < 0:
                end_B -= start_B
                start_B = 0
            elif end_B > len(d2):
                start_B -= end_B - len(d2)
                end_B = len(d2)
            assert end_B - start_B == L
            d2[start_b:end_b] *= nb_beta(sigma, L, d2[start_B:end_B])
        res.append(d2)
    return res

def _signal_smoothing(mpl_signal, mpl_range, nb_factor=0.1, wavelet = "bior3.5", merge_range = 5, plot_bool = False):
    # apply discrete wavelet smoothing to 
    even_len_bool = False
    mpl_signal = np.nan_to_num(mpl_signal, nan=0.0)
    if mpl_signal.size % 2 == 0:
        mpl_signal = mpl_signal[:-1]
        mpl_range  = mpl_range[:-1]
        even_len_bool = True
    coeffs_n = pywt.wavedec(mpl_signal, wavelet, mode = 'constant')
    approx_n = coeffs_n[0]
    details_n = coeffs_n[1:]

    new_detail = neigh_block(details_n, len(mpl_signal), nb_factor)
    smoothed_signal= pywt.waverec([approx_n] + new_detail, wavelet)[1:]

    smoothed_multiplier   = scipy.stats.norm.cdf(mpl_range, merge_range, 1)
    unsmoothed_multiplier = 1 - smoothed_multiplier
    combined_signal = smoothed_signal * smoothed_multiplier + mpl_signal * unsmoothed_multiplier

    if plot_bool:
        plt.figure(figsize=(15,24))
        plot_dwt(details_n, approx_n, color="red", alpha=0.5)
        plot_dwt(new_detail, approx_n, color="green", alpha=0.5)
        plt.show()

    if even_len_bool:
        combined_signal = np.concatenate((combined_signal, [combined_signal[-1]]))

    return combined_signal


def signal_smoothing(mpl_signal, mpl_range, nb_factor=0.1, wavelet = "bior3.5", merge_range = 5, plot_bool = False):
    mpl_signal = np.array(mpl_signal)
    if mpl_signal.ndim == 1:
        return _signal_smoothing(mpl_signal, mpl_range, nb_factor=nb_factor, wavelet = wavelet, merge_range = merge_range, plot_bool = plot_bool)
    elif mpl_signal.ndim == 2:
        combined_signal_list = []
        for i in range(mpl_signal.shape[0]):
            combined_signal_list.append(_signal_smoothing(mpl_signal[i], mpl_range, nb_factor=nb_factor, wavelet = wavelet, merge_range = merge_range, plot_bool = plot_bool))
        return np.array(combined_signal_list)
    else:
        raise ValueError('mpl_signal has to be an 1D array or a 2D array')
    

def _signal_smoothing_snr_selection(mpl_signal, mpl_snr, mpl_range, snr_var = 5, smooth_level = 100, wavelet = "bior3.5", plot_bool = False):
    # apply discrete wavelet smoothing to 
    even_len_bool = False
    mpl_signal = np.nan_to_num(mpl_signal, nan=0.0)
    if mpl_signal.size % 2 == 0:
        mpl_signal = mpl_signal[:-1]
        mpl_range  = mpl_range[:-1]
        even_len_bool = True
    coeffs_n = pywt.wavedec(mpl_signal, wavelet, mode = 'constant')
    approx_n = coeffs_n[0]
    details_n = coeffs_n[1:]

    snr_factor = np.copy(mpl_snr)
    snr_factor[snr_factor<0] = 0
    quality_index = np.nanmin(np.where(snr_factor<snr_var)[0])
    snr_factor = snr_factor/(snr_var * smooth_level)
    snr_factor[snr_factor>1] = 1
    ones_factor = np.ones(snr_factor.shape)

    smoothed_multiplier   = scipy.stats.norm.cdf(mpl_range, mpl_range[quality_index], 1)
    unsmoothed_multiplier = 1 - smoothed_multiplier
    snr_factor = snr_factor * smoothed_multiplier + ones_factor * unsmoothed_multiplier


    plt.figure(figsize=(15,4))
    plt.plot(mpl_range, snr_factor)

    new_detail = []

    for i in range(len(details_n)):
        detail_x = range(len(details_n[i]))
        #print(mpl_range/np.nanmax(mpl_range))
        a_snr_factor = np.interp(detail_x/np.nanmax(detail_x), mpl_range/np.nanmax(mpl_range), snr_factor)
        a_new_detail = np.array(details_n[i]) * np.array(a_snr_factor)
        new_detail.append(a_new_detail)

    if plot_bool:
        plt.figure(figsize=(15,24))
        plot_dwt(details_n, approx_n, color="red", alpha=0.5)
        plot_dwt(new_detail, approx_n, color="green", alpha=0.5)
        plt.show()

    
    smoothed_signal= pywt.waverec([approx_n] + new_detail, wavelet)[1:]


    return smoothed_signal

    
def signal_smoothing_snr_selection(mpl_signal, mpl_snr, mpl_range, wavelet = "bior3.5", merge_range = 5, plot_bool = False):
    mpl_signal = np.array(mpl_signal)
    if mpl_signal.ndim == 1:
        return _signal_smoothing_snr_selection(mpl_signal, mpl_snr, mpl_range, wavelet = "bior3.5", merge_range = 5, plot_bool = False)
    elif mpl_signal.ndim == 2:
        combined_signal_list = []
        for i in range(mpl_signal.shape[0]):
            combined_signal_list.append(_signal_smoothing_snr_selection(mpl_signal, mpl_snr, mpl_range, wavelet = "bior3.5", merge_range = 5, plot_bool = False))
        return np.array(combined_signal_list)
    else:
        raise ValueError('mpl_signal has to be an 1D array or a 2D array')

def slope_inversion(nrb, mpl_range, calibration_range, slope_start, slope_end, S1 , S2 = molecular_lidar_ratio, beta2=None, wavelength = 0.532):
    '''
    Slope inverison follows:
    Klett, J. D. (1981). Stable analytical inversion solution for processing lidar returns. Applied optics, 20(2), 211-220.
    '''
    lbNRB = np.log(nrb)
    lbNRB = np.nan_to_num(lbNRB, nan=0.0)
    selected_lnNRB = lbNRB[np.logical_and(mpl_range >= slope_start, mpl_range < slope_end)]
    selected_range = mpl_range[np.logical_and(mpl_range >= slope_start, mpl_range < slope_end)]

    slope, intercept, r, p_value, std_err = scipy.stats.linregress(selected_range, selected_lnNRB)
    slope_ext_coeff = -0.5 * slope
    
    if beta2 is None:
        beta2 = rayleigh_bcksca_coeff(wavelength, mpl_range)
    
    beta2_c_coeff = np.interp(calibration_range, mpl_range, beta2)
    ext2_c_coeff = beta2_c_coeff * molecular_lidar_ratio
    aerosol_bcksca  = (slope_ext_coeff - ext2_c_coeff)/S1
    slope_bcksca_coeff = aerosol_bcksca + beta2_c_coeff

    return slope_ext_coeff, slope_bcksca_coeff, slope, intercept, r, p_value, std_err


def iterative_slope_inversion(nrb, mpl_range, calibration_range, slope_start, slope_end, beta2, S1, S2 = molecular_lidar_ratio, beta1 = 0, iteration_count = 0, max_iteration = 100, show_info = False):
    """
    S1: Aerosol lidar ratio
    S2: Molecular lidar ratio 8pi/3
    """

    lbNRB = np.log(nrb)
    lbNRB = np.nan_to_num(lbNRB, nan=0.0)
    selected_lnNRB = lbNRB[np.logical_and(mpl_range >= slope_start, mpl_range < slope_end)]
    selected_beta2 = beta2[np.logical_and(mpl_range >= slope_start, mpl_range < slope_end)]
    selected_range = mpl_range[np.logical_and(mpl_range >= slope_start, mpl_range < slope_end)]
    slope, intercept, r, p_value, std_err = scipy.stats.linregress(selected_range, selected_lnNRB)

    slope_beta2, intercept_beta2, r_beta2, p_value_beta2, std_err_beta2= scipy.stats.linregress(selected_range, selected_beta2)

    calibration_beta2 = np.interp(calibration_range, mpl_range, beta2)
    total_beta = calibration_beta2 + beta1
    first_term = np.divide(slope_beta2, total_beta)
    slope_ext_coeff = -0.5 * (slope-first_term)
    lidar_ratio = (calibration_beta2/total_beta) * S2 + (beta1/total_beta) * S1
    slope_bcksca_coeff = slope_ext_coeff/lidar_ratio
    next_beta1 = slope_bcksca_coeff - calibration_beta2
    next_lidar_ratio = (calibration_beta2/slope_bcksca_coeff) * S2 + (next_beta1/slope_bcksca_coeff) * S1

    if show_info:
        print(f'iteration #{iteration_count}')
        print(f'current total lidar ratio is {lidar_ratio}, next lidar ratio is {next_lidar_ratio}, difference is {np.absolute(next_lidar_ratio - lidar_ratio)}')

    if (np.absolute(next_lidar_ratio - lidar_ratio) > 0.1) and (iteration_count < max_iteration) and next_beta1 >=0:
        return iterative_slope_inversion(nrb, mpl_range, calibration_range, slope_start, slope_end, beta2, S1, S2 = molecular_lidar_ratio, beta1 = next_beta1, iteration_count=iteration_count+1, max_iteration = max_iteration, show_info = show_info)
    
    else:
        #print('end iteration!')
        return slope_ext_coeff, slope_bcksca_coeff, slope, intercept, r, p_value, std_err
    