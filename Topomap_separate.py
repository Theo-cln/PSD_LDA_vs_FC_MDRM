import numpy as np
import mne

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mne.defaults import _EXTRAPOLATE_DEFAULT, _BORDER_DEFAULT

from mne.io.meas_info import Info, _simplify_info
from mne.viz import topomap
from mne.channels.layout import _find_topomap_coords



"""initial code"""
def add_colorbar(ax, im, cmap, side="right", pad=.05, title=None,
                  format=None, size="5%"):
    """Add a colorbar to an axis."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size=size, pad=pad)
    cbar = plt.colorbar(im, cax=cax, format=format)


    return cbar, cax


def plot_topomap_data_viz(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,
                 res=64, axes=None, names=None, show_names=False, mask=None,
                 mask_params=None, outlines='head',
                 contours=6, image_interp='linear', show=True,
                 onselect=None, extrapolate=_EXTRAPOLATE_DEFAULT,
                 sphere=None, border=_BORDER_DEFAULT,
                 ch_type='eeg',freq='10',Stat_method='R_square signed', phase_name='phase ...', frequency_name=None):
    """Plot a topographic map as image.

    Parameters
    ----------
    data : array, shape (n_chan,)
        The data values to plot.
    pos : array, shape (n_chan, 2) | instance of Info
        Location information for the data points(/channels).
        If an array, for each data point, the x and y coordinates.
        If an Info object, it must contain only one data type and
        exactly ``len(data)`` data channels, and the x/y coordinates will
        be inferred from this Info object.
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | None
        Colormap to use. If None, 'jet' is used for all positive data,
        otherwise defaults to 'RdBu_r'.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib plot
        format string (e.g., 'r+' for red plusses). If True (default), circles
        will be used.
    res : int
        The resolution of the topomap image (n pixels along each side).
    axes : instance of Axes | None
        The axes to plot to. If None, the current axes will be used.
    names : list | None
        List of channel names. If None, channel names are not plotted.
    %(topomap_show_names)s
        If ``True``, a list of names must be provided (see ``names`` keyword).
    mask : ndarray of bool, shape (n_channels, n_times) | None
        The channels to be marked as significant at a given time point.
        Indices set to ``True`` will be considered. Defaults to None.
    mask_params : dict | None
        Additional plotting parameters for plotting significant sensors.
        Default (None) equals::

           dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)
    %(topomap_outlines)s
    contours : int | array of float
        The number of contour lines to draw. If 0, no contours will be drawn.
        If an array, the values represent the levels for the contours. The
        values are in µV for EEG, fT for magnetometers and fT/m for
        gradiometers. Defaults to 6.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    show : bool
        Show figure if True.
    onselect : callable | None
        Handle for a function that is called when the user selects a set of
        channels by rectangle selection (matplotlib ``RectangleSelector``). If
        None interactive selection is disabled. Defaults to None.
    %(topomap_extrapolate)s

        .. versionadded:: 0.18
    %(topomap_sphere)s
    %(topomap_border)s
    %(topomap_ch_type)s

    Returns
    -------
    im : matplotlib.image.AxesImage
        The interpolated data.
    cn : matplotlib.contour.ContourSet
        The fieldlines.
    """
    sphere = topomap._check_sphere(sphere)
    return _plot_topomap_test(data, pos, vmin, vmax, cmap, sensors, res, axes,
                         names, show_names, mask, mask_params, outlines,
                         contours, image_interp, show,
                         onselect, extrapolate, sphere=sphere, border=border,
                         ch_type=ch_type,freq=freq,Stat_method=Stat_method, phase_name=phase_name, frequency_name=frequency_name)[:2]
def _plot_topomap_test(data, pos, vmin=None, vmax=None, cmap=None, sensors=True,res=64, axes=None, names=None, show_names=False, mask=None,mask_params=None, outlines=None,contours=6, image_interp='bilinear', show=True,onselect=None, extrapolate=_EXTRAPOLATE_DEFAULT, sphere=None,border=0, ch_type='eeg',freq = '10',Stat_method = 'R square signed', phase_name='phase ...', frequency_name=None):
    data = np.asarray(data)
    top = cm.get_cmap('YlOrRd_r', 128) # r means reversed version
    bottom = cm.get_cmap('YlGnBu_r', 128)
    newcolors2 = np.vstack((bottom(np.linspace(0, 1, 128)),top(np.linspace(1, 0, 128))))
    double = ListedColormap(newcolors2, name='double')
    if isinstance(pos, Info):  # infer pos from Info object
        picks = topomap._pick_data_channels(pos, exclude=())  # pick only data channels
        pos: dict = topomap.pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = pos.get_channel_types(picks=None, unique=True)  #topomap._get_channel_types(pos, unique=True)
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.io.pick.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object (%s) and "
                             "the data array (%s) do not match. "
                             % (len(pos['chs']), data.shape[0]) + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            picks = topomap._pair_grad_sensors(pos, topomap_coords=False)
            pos = topomap._find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = topomap._merge_ch_data(data[picks], ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)

    extrapolate = topomap._check_extrapolate(extrapolate, ch_type)
    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = topomap._setup_vmin_vmax(data, vmin, vmax, norm)


    outlines = topomap._make_head_outlines(sphere, pos, outlines, (0., 0.))
    assert isinstance(outlines, dict)

    ax = axes if axes else plt.gca()
    topomap._prepare_topomap(pos, ax)

    mask_params = topomap._handle_default('mask_params', mask_params)
    print(mask_params)
    # find mask limits
    extent, Xi, Yi, interp = topomap._setup_interp(
        pos, res, "linear", extrapolate, outlines, border) # circle <==> extrapolate
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = topomap._get_patch(outlines, extrapolate, interp, ax)
    # plot interpolated map
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=extent)
    cbar,cax = add_colorbar(ax, im, cmap, side="right", pad=.1, title=None,
                      format=None, size="5%")
    cbar.ax.tick_params(labelsize=18)
    #cbar.set_label('MI-Rest/Rest', rotation=270,labelpad = 15)
    cbar.set_label('Node strength difference', rotation=270,labelpad = 15)
    #cbar.set_label('Cluster size evaluate at p<0.01', rotation=270,labelpad = 20)
    #cbar.set_label('Rho Spearman at p<0.05', rotation=270,labelpad = 15)
    #ax.set_title(freq +'(Hz)',fontsize = 'large')
    # ax.set_title(freq,fontsize = 'large')
    ax.set_title(f'Topomap {phase_name} {frequency_name} Hz')
    # gh-1432 had a workaround for no contours here, but we'll remove it
    # because mpl has probably fixed it
    linewidth = mask_params['markeredgewidth']
    cont = True
    if isinstance(contours, (np.ndarray, list)):
        pass
    elif contours == 0 or ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        cont = None  # can't make contours for constant-valued functions
    if cont:
        cont = ax.contour(Xi, Yi, Zi, contours, colors='k',
                          linewidths=linewidth / 2.)

    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T
    if sensors is not False and mask is None:
        topomap._topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        topomap._topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        ax.set_title(f'Topomap {phase_name}')


    if isinstance(outlines, dict):
        topomap._draw_outlines(ax, outlines)

    if show_names:
        if names is None:
            raise ValueError("To show names, a list of names must be provided"
                             " (see `names` keyword).")
        if show_names is True:
            def _show_names(x):
                return x
        else:
            _show_names = show_names
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = _show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', fontsize=9,fontweight='bold')

    plt.subplots_adjust(top=.95)

    if onselect is not None:
        lim = ax.dataLim
        x0, y0, width, height = lim.x0, lim.y0, lim.width, lim.height
        ax.RS = RectangleSelector(ax, onselect=onselect)
        ax.set(xlim=[x0, x0 + width], ylim=[y0, y0 + height])
    topomap.plt_show(show)
    return im, cont, interp

def topo_plot(Rsquare,freq,electrodes,fs,Stat_method,vmin,vmax,axes=None, phase_name='phase ...', frequency_name=None):
    size_dim = mne.channels.make_standard_montage('standard_1020')
    # if len(electrodes) >32:
    #     biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    # else:
    biosemi_montage_inter = mne.channels.make_standard_montage('standard_1020')
    print(biosemi_montage_inter.ch_names)
    ind = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel in electrodes]
    ind_notin = [i for (i, channel) in enumerate(biosemi_montage_inter.ch_names) if channel not in electrodes]
    biosemi_montage = biosemi_montage_inter.copy()
    # Keep only the desired channels
    Chan_names = [biosemi_montage_inter.ch_names[x] for x in ind]
    biosemi_montage.ch_names = [biosemi_montage_inter.ch_names[x] for x in ind] + [biosemi_montage_inter.ch_names[x] for x in ind_notin]
    kept_channel_info = [biosemi_montage_inter.dig[x+3] for x in ind] + [biosemi_montage_inter.dig[x+3] for x in ind_notin]
    # Keep the first three rows as they are the fiducial points information
    biosemi_montage.dig = biosemi_montage_inter.dig[0:3]+kept_channel_info
        #biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    n_channels = len(biosemi_montage.ch_names)

    # first we obtain the 3d positions of selected channels
    chs = ['Iz', 'CPz', 'TP9', 'TP10']
    pos = np.stack([size_dim.get_positions()['ch_pos'][ch] for ch in chs])

    print(pos)
    # now we calculate the radius from T7 and T8 x position
    # (we could use Oz and Fpz y positions as well)
    print("Radius")
    radius = np.abs(pos[[2, 3], 0]).mean()-0.02
    print(radius)
        # modify the y position of each electrode
    for ch_name in range(len(electrodes)):
        # modify y position of electrode
        y = biosemi_montage.get_positions()['ch_pos'][electrodes[ch_name]][1]

        y_shifted = 1.1*y - 0.008  # add 0.1 units to y position
        biosemi_montage.get_positions()['ch_pos'][electrodes[ch_name]][1] = y_shifted

        x = biosemi_montage.get_positions()['ch_pos'][electrodes[ch_name]][0]

        x_shifted = x *1.15  # add 0.1 units to y position
        biosemi_montage.get_positions()['ch_pos'][electrodes[ch_name]][0] = x_shifted


    fake_info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=fs/2,
                                ch_types='eeg')

    rng = np.random.RandomState(0)
    data = rng.normal(size=(n_channels, 1)) * 1e-6
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(biosemi_montage)

    # then we obtain the x, y, z sphere center this way:
    # x: x position of the Oz channel (should be very close to 0)
    # y: y position of the T8 channel (should be very close to 0 too)
    # z: average z position of Oz, Fpz, T7 and T8 (their z position should be the
    #    the same, so we could also use just one of these channels), it should be
    #    positive and somewhere around `0.03` (3 cm)
    x = pos[0, 0]
    y = pos[-1, 1]
    z = pos[:, -1].mean()
    sizer = np.zeros((n_channels))
    print(x)
    print(y)
    print(z)
    print(n_channels)
    for i in range(n_channels):
        for j in range(len(electrodes)):
            if(biosemi_montage.ch_names[i]==electrodes[j]):
                sizer[i] = Rsquare[j, freq]
    freq = str(freq)
    top = cm.get_cmap('YlOrRd_r', 128) # r means reversed version
    bottom = cm.get_cmap('YlGnBu_r', 128)
    newcolors2 = np.vstack((bottom(np.linspace(0, 1, 128)),top(np.linspace(1, 0, 128))))
    double = ListedColormap(newcolors2, name='double')
    #Chan_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', '', '', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', '', '', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', '', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', '', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', '']
    #plot_topomap_data_viz(sizer, fake_evoked.info,sensors = False,names = Chan_names,show_names = True,res = 500,mask_params = dict(marker='', markerfacecolor='w', markeredgecolor='k',linewidth=0, markersize=0),contours = 0,image_interp='nearest',show=True, extrapolate='head',border = 0,cmap='jet',freq = freq,Stat_method=Stat_method,vmin = vmin,vmax=vmax,axes=axes)
    plot_topomap_data_viz(sizer, fake_evoked.info,sensors = False,names = Chan_names,show_names = True,res = 500,mask_params = dict(marker='', markerfacecolor='w', markeredgecolor='k',linewidth=0, markersize=0),contours = 1,image_interp='linear',show=True, extrapolate='head',cmap='bwr',freq = freq,Stat_method=Stat_method,vmin = vmin,vmax=vmax,axes=axes, phase_name=phase_name, frequency_name=frequency_name)
