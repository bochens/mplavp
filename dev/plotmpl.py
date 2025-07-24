import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap

def _make_lidar_colormap(cmap_str, under_color=np.array([0, 0, 0, 1]), over_color=np.array([1, 1, 1, 1])):
    cmap = cm.get_cmap(cmap_str, 256)
    newcolors = cmap(np.linspace(0, 1, 256))
    newcolors[:1, :] = under_color  # black
    newcolors[-1:,:] = over_color # white
    lidar_cmap = ListedColormap(newcolors)
    return lidar_cmap

##### lidar colormap #####
gnuplot2     = cm.get_cmap('gnuplot2', 256)
nipy_spectral= cm.get_cmap('nipy_spectral', 256)
lidar_jet    = _make_lidar_colormap('jet')
lidar_gist   = _make_lidar_colormap('gist_ncar')
lidar_turbo  = _make_lidar_colormap('turbo')
inferno      = cm.get_cmap('inferno', 256)

##### default parameters #####
default_figure_size = (7.2, 4.8)

def _get_fig_and_ax(fig, ax):
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=default_figure_size)
    elif fig is None and ax is not None:
        fig = ax.get_figure()
    elif fig is not None and ax is None:
        ax = fig.add_subplot(111)
    return fig, ax

def _format_datetime(datetime_arr):
    return_list = []
    formatted_str = ""

    date_and_time_format = "%H:%M:%S\n%m-%d-%Y"
    time_format = '%H:%M:%S'
    
    for idx, x in enumerate(datetime_arr):
        if idx == 0:
            formatted_str += x.item().strftime(date_and_time_format)
        else:
            previous_x = datetime_arr[idx-1]
            if previous_x.astype('datetime64[D]') == x.astype('datetime64[D]'):
                formatted_str += x.item().strftime(time_format)
            else:
                formatted_str += x.item().strftime(date_and_time_format)
        return_list.append(formatted_str)
        formatted_str = ""
    return np.array(return_list)

def plot_mpl_2d_timeseries(mpl_datetime, mpl_range, mpl_2d_data, fig=None, ax=None, range_max = 20, vmin=0, vmax=1, color_map = lidar_jet, colorbar_bool = True, colorbar_label=None, colorbar_norm = 'linear', x_tick_number = None):
    """
        Plot a 2D time series with a colorbar.
        
        Parameters:
        -----------
        mpl_datetime : array-like
            Array of datetime objects representing the time dimension.
        mpl_range : array-like
            Array representing the vertical range dimension (e.g., altitude).
        mpl_2d_data : 2D array
            Data to be plotted, with time as the first axis and range as the second.
        fig : Matplotlib figure object, optional
            The figure to plot on. If None, a new figure will be created.
        ax : Matplotlib axis object, optional
            The axis to plot on. If None, a new axis will be created.
        range_max : float, optional
            Maximum y-range to display in kilometers. Default is 20.
        vmin : float, optional
            Minimum value for the colormap. If log scaling is used, vmin must be > 0.
        vmax : float, optional
            Maximum value for the colormap.
        color_map : str or Colormap, optional
            Colormap to use for the plot. Default is 'viridis'.
        colorbar_bool : bool, optional
            Whether to display a colorbar. Default is True.
        colorbar_label : str, optional
            Label for the colorbar. Default is None (no label).
        colorbar_norm : str, optional
            'linear' or 'log' scaling for the colormap. Default is 'linear'.
        x_tick_number : int, optional
            Number of ticks on the x-axis. If None, it will be determined automatically.
        
        Returns:
        --------
        handle : QuadMesh
            The object returned by `pcolormesh`, useful for further customization (e.g., adding a colorbar).
    """
    fig, ax = _get_fig_and_ax(fig, ax)
    ax.set_facecolor('k')
    time_x  = np.arange(mpl_datetime.shape[0]+1)     # time dimension edge for pcolormesh
    y_resolution = mpl_range[1]-mpl_range[0]         # y resolution
    range_y = np.append(mpl_range-y_resolution/2, mpl_range[-1]+y_resolution/2) # pixel edge height 

    if vmin <= 0:
        positive_data = mpl_2d_data[mpl_2d_data > 0]
        if positive_data.size > 0:  # Check if there are any positive values
            log_vmin = np.nanmin(positive_data)
        else:
            log_vmin = np.nan  # Default to 1 (or any sensible minimum value) if no positive values
    else:
        log_vmin = vmin

    if colorbar_norm == "linear":
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    elif (colorbar_norm == "log") or (colorbar_norm == "Log"):
        norm = colors.LogNorm(vmin=log_vmin, vmax=vmax)

    handle = ax.pcolormesh(time_x, range_y, np.transpose(mpl_2d_data), cmap = color_map, shading='auto' , norm=norm)
    ax.set_ylim(0,range_max)

    if colorbar_bool:
        cbar = fig.colorbar(handle, ax=ax, extend='both') # draw colorbar
        if colorbar_label:
            cbar.set_label(colorbar_label)

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    # x ticking and label
    if x_tick_number is None:
        x_tick_number = int(width) # will automatically determine how many ticks and lebels to use depending on the figure width

    x_tick_positions_availiable = time_x[:-1] + 0.5
    x_tick_position_selector = np.linspace(x_tick_positions_availiable[0], x_tick_positions_availiable[-1], num = x_tick_number).astype(int)
    ax.set_xticks(x_tick_positions_availiable[x_tick_position_selector])
    formatted_time_label_string = _format_datetime(mpl_datetime[x_tick_position_selector])
    ax.set_xticklabels(formatted_time_label_string)

    ax.set_ylabel('Altitude AGL (km)')
    return handle

def plot_mpl_1d_timeseries(mpl_datetime, mpl_1d_data, fig=None, ax=None, ylabel = None, tick_number = None, color = '#1f77b4'):
    fig, ax = _get_fig_and_ax(fig, ax)

    time_x = np.arange(mpl_datetime.size)
    handle = ax.plot(time_x, mpl_1d_data, color = color)

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height

    # x ticking and label
    if tick_number is None:
        tick_number = int(width) # will automatically determine how many ticks and lebels to use depending on the figure width

    x_tick_positions_availiable = time_x
    x_tick_position_selector = np.linspace(time_x[0], time_x[-1], num = tick_number).astype(int)
    ax.set_xticks(x_tick_positions_availiable[x_tick_position_selector])
    formatted_time_label_string = _format_datetime(mpl_datetime[x_tick_position_selector])
    ax.set_xticklabels(formatted_time_label_string)

    ax.set_xlabel('time')

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
#def profile_plot

