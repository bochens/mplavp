import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pympl import PyMPL
import pywt
from numba import jit
from numba.core.errors import NumbaDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# Micropulse Lidar Layer Finding Based on Continuous Wavelet Transform


def find_ridge_lines(widths, all_local_maxima, gap_treshold = 1):
    ridge_lines = [] # each ridge line is a list containing a list of indexes from the scale where the ridge line is initialized to smallest scale
    gap_numbers = []
    
    for j in range(len(widths)-1, -1, -1):
        distance_treshold = (17*(widths[j]) //20)*2+1
        if j == len(widths)-1:
            new_next_maximas = np.array(all_local_maxima[j])
        
        for a_maxima in new_next_maximas:
            ridge_lines.append({j: a_maxima})
            gap_numbers.append(0)

        new_next_maximas = np.array(all_local_maxima[j-1])
        for ridge_i, a_ridge in enumerate(ridge_lines):
            if (j in a_ridge) & (j != 0):
                next_ridge_index = np.argmin(np.abs(a_ridge[j]-np.array(all_local_maxima[j-1])))
                distance = np.abs(a_ridge[j]-all_local_maxima[j-1][next_ridge_index])
                if distance > distance_treshold:
                    gap_numbers[ridge_i] = gap_numbers[ridge_i] + 1

                if gap_numbers[ridge_i] <= gap_treshold:
                    ridge_lines[ridge_i].update({j-1:all_local_maxima[j-1][next_ridge_index]})
                    new_next_maximas = np.delete(new_next_maximas, new_next_maximas == all_local_maxima[j-1][next_ridge_index])

    return ridge_lines


def select_long_ridge_lines(ridge_lines, cwtmatr, length_treshold):
    selected_ridge_lines = []
    selected_ridge_means = []
    selected_ridge_value = []
    for i, a_ridge_line in enumerate(ridge_lines):
        if (len(a_ridge_line) >= length_treshold):
            average_list = []
            for key, value in a_ridge_line.items():
                average_list.append(cwtmatr[key][value])

            selected_ridge_value.append(average_list[0])
            selected_ridge_lines.append(a_ridge_line)
            selected_ridge_means.append(np.nanmean(average_list))

    selected_ridge_lines = np.array(selected_ridge_lines)
    selected_ridge_value = np.array(selected_ridge_value)
    selected_ridge_means = np.array(selected_ridge_means)

    return selected_ridge_lines, selected_ridge_value, selected_ridge_means

@jit
def sliding_window_local_maxmin(data, scale):
    local_maxima = []
    local_minima = []
    k = 17*scale //10
    # loop over the data, excluding the border values where the window cannot fit
    for i in range(0, k):
        if data[i] == np.max(data[0:i+k+1]):
            local_maxima.append(i)
        if data[i] == np.min(data[0:i+k+1]):
            local_minima.append(i)
    for i in range(k, len(data) - k):
        # check if the current point is a local maximum
        if data[i] == np.max(data[i-k:i+k+1]):
            local_maxima.append(i)
        if data[i] == np.min(data[i-k:i+k+1]):
            local_minima.append(i)
    for i in range(len(data) - k, len(data)):
        if data[i] == np.max(data[0:i+k+1]):
            local_maxima.append(i)
        if data[i] == np.min(data[0:i+k+1]):
            local_minima.append(i)

    return local_maxima, local_minima

@jit
def get_all_local_maxmin(cwtmatr, widths):
    all_local_maxima = []
    all_local_minima = []
    for scale in widths:
        data = cwtmatr[scale-1]
        local_maxima, local_minima = sliding_window_local_maxmin(data, scale)
        all_local_maxima.append(local_maxima)
        all_local_minima.append(local_minima)
    
    return all_local_maxima, all_local_minima


def cwt_find_cloud_edges(nrb_profile, scale_num = 31, length_treshold = 30, value_treshold = 2):
    widths = np.arange(1, scale_num)

    cwtmatr, freqs = pywt.cwt(nrb_profile, widths, 'gaus1')
    all_local_maxima, all_local_minima = get_all_local_maxmin(cwtmatr, widths)

    ridge_lines  = find_ridge_lines(widths, all_local_maxima, gap_treshold = 1)
    trough_lines = find_ridge_lines(widths, all_local_minima, gap_treshold = 1)

    selected_ridge_lines, selected_ridge_value, selected_ridge_means  = select_long_ridge_lines(ridge_lines,  cwtmatr, length_treshold)
    high_signal_ridge_lines = selected_ridge_lines[np.array(selected_ridge_value)>=value_treshold]
    selected_trough_lines, selected_trough_value, selected_trough_means = select_long_ridge_lines(trough_lines, cwtmatr, length_treshold)
    high_signal_trough_lines = selected_trough_lines[np.array(selected_trough_value)<=-value_treshold]


    high_signal_ridge_lines_list = np.array([np.nanmean(list(d.values())[:1]) for d in high_signal_ridge_lines])
    high_signal_trough_lines_list = np.array([np.nanmean(list(d.values())[:1]) for d in high_signal_trough_lines])

    return high_signal_ridge_lines_list, high_signal_trough_lines_list

cloud_pixel_number_treshold = 2

@jit
def has_consecutive_values(arr, idx1, idx2, contain_cloud_treshold, cloud_pixel_number_treshold):
    count = 0  # Counter for consecutive values >= 1
    
    # Iterate through the array between the two indices
    for val in arr[idx1 + 1:idx2]:
        if val >= contain_cloud_treshold:
            count += 1  # Increment the counter
            
            # If there are at least two consecutive values, return True
            if count >= cloud_pixel_number_treshold:
                return True
        else:
            count = 0  # Reset the counter if the value is less than 1
    
    return False  # Return False if no such consecutive values are found

@jit
def find_real_bottoms(cloud_bottoms, cloud_tops, backscatter, contain_cloud_treshold, cloud_pixel_number_treshold):
    real_bottoms = []
    i = 0
    while i < (len(cloud_bottoms)):
        if i < (len(cloud_bottoms)-1) and (not np.any(np.logical_and(cloud_tops>cloud_bottoms[i], cloud_tops<cloud_bottoms[i+1]))):
            # if there are two consecutive cloud bottoms and no cloud top in between
            if has_consecutive_values(backscatter, cloud_bottoms[i], cloud_bottoms[i+1], contain_cloud_treshold, cloud_pixel_number_treshold):
                # if there is acturally cloud between the two consecutive cloud bottoms, use the lower bottom and skip the higher one.
                real_bottoms.append(cloud_bottoms[i])
                i = i+1
            else:
                # the next cloud bottom is the real one.
                pass
        elif not np.any(cloud_tops>cloud_bottoms[i]):
                # when there is no more cloud top above this cloud bottom, skip this one.
                pass
        else:
            real_bottoms.append(cloud_bottoms[i])
        
        i = i+1
    real_bottoms = np.array(real_bottoms)
    return real_bottoms

@jit
def find_real_clouds_aux(cloud_bottoms, cloud_tops, backscatter, contain_cloud_treshold, cloud_pixel_number_treshold):
    # Sort lists for orderly checking
    cloud_bottoms = np.sort(cloud_bottoms)
    cloud_tops = np.sort(cloud_tops)

    profile_height = backscatter.size

    real_bottoms = find_real_bottoms(cloud_bottoms, cloud_tops, backscatter, contain_cloud_treshold, cloud_pixel_number_treshold)
    real_tops    = np.sort(profile_height-find_real_bottoms(np.sort(profile_height-cloud_tops), np.sort(profile_height-cloud_bottoms), np.flip(backscatter), contain_cloud_treshold, cloud_pixel_number_treshold))
    # flip around now tops are bottom and bottoms are tops
    
    return real_bottoms, real_tops

@jit
def find_indices_beyond_threshold(array, threshold):
    first_index = None
    last_index = None

    for index, value in enumerate(array):
        if value > threshold:
            if first_index is None:
                first_index = index
            last_index = index

    return first_index, last_index

@jit
def find_real_clouds(cloud_bottoms, cloud_tops, backscatter, edge_treshold = 0.5, contain_cloud_treshold = 1, cloud_pixel_number_treshold = 2):
    real_bottoms, real_tops = find_real_clouds_aux(cloud_bottoms, cloud_tops, backscatter, contain_cloud_treshold, cloud_pixel_number_treshold)
    min_length = min(len(real_bottoms), len(real_tops))  # Find the minimum length between the two lists

    real_bottoms = real_bottoms[:min_length]  # Take the first min_length elements from list1
    real_tops    = real_tops[:min_length]  # Take the first min_length elements from list2

    actural_bottoms = []
    actural_tops    = []

    for i in range(min_length):
        if real_bottoms[i] != real_tops[i] and has_consecutive_values(backscatter, real_bottoms[i], real_tops[i], contain_cloud_treshold, cloud_pixel_number_treshold):
            an_actural_bottom, an_actural_top = find_indices_beyond_threshold(backscatter[real_bottoms[i]:real_tops[i]], edge_treshold)
            
            actural_bottoms.append(an_actural_bottom+real_bottoms[i])
            actural_tops.append(an_actural_top+real_bottoms[i])

    actural_bottoms = np.array(actural_bottoms)
    actural_tops    = np.array(actural_tops)
    return actural_bottoms, actural_tops


def cwt_cloud_mask(data, data_range, range_maximum = 10, scale_num = 31, length_treshold = 30, value_treshold = 0.5, edge_treshold = 0.5, contain_cloud_treshold = 0.75, cloud_pixel_number_treshold = 2, not_a_cloud_treshold = 0.3):
    cloud_mask = np.zeros(data.shape)

    all_cloud_bottoms = []
    all_cloud_tops    = []

    for profile_i in range(data.shape[0]):

        nrb_profile = data[profile_i][data_range<range_maximum]
        ridge_lines, trough_lines = cwt_find_cloud_edges(nrb_profile, scale_num = scale_num, length_treshold = length_treshold, value_treshold = value_treshold)
        cloud_bottoms, cloud_tops = find_real_clouds(trough_lines.astype(int), ridge_lines.astype(int), nrb_profile, edge_treshold = edge_treshold, contain_cloud_treshold = contain_cloud_treshold, cloud_pixel_number_treshold = cloud_pixel_number_treshold)
        all_cloud_bottoms.append(cloud_bottoms)
        all_cloud_tops.append(cloud_tops)

        for i in range(cloud_bottoms.size):
            cloud_mask[profile_i, cloud_bottoms[i] : cloud_tops[i]+1] = 1

        cloud_mask[profile_i][data[profile_i]<not_a_cloud_treshold] = 0

    return cloud_mask, all_cloud_bottoms, all_cloud_tops