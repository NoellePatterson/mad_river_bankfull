import geopandas as gpd
import pandas as pd
import numpy as np
import ast
from numpy import nan
from scipy import stats
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def plot_longitudinal_profile():
    # Plot full longitudinal profile to search for features such as knickpoints
    all_widths_df = pd.read_csv('data_outputs/all_widths.csv')
    # Extract and detrend thalweg for plotting
    thalweg_distances = all_widths_df['thalweg_distance']
    total_distance = 0
    all_distances = []
    for distance in thalweg_distances:
        total_distance += distance
        all_distances.append(total_distance)
    x_vals = all_distances
    y_vals = all_widths_df['thalweg_elev']
    x = np.array(x_vals).reshape(-1, 1)
    y = np.array(y_vals)
    linear_model = LinearRegression().fit(x, y)
    slope = linear_model.coef_
    intercept = linear_model.intercept_
    # plot results of linear model
    y_linear = linear_model.predict(x)
    fit_slope = slope*x
    # Identify zones of high slope
    marker_x = []
    marker_y = []
    transect_knickpoints = []
    for y_index, y_val in enumerate(y_vals):
        if y_index < len(y_vals) - 1:
            if y_val - y_vals[y_index + 1] > 3: # for single step changes greater than 3 meters:
                marker_x.append(x_vals[y_index])
                marker_y.append(y_vals[y_index])
                transect_knickpoints.append(y_index)
    # Plot longitudinal profile
    fig, ax = plt.subplots()
    plt.xlabel('Thalweg distance from upstream (m)')
    plt.ylabel('Thalweg elevation ASL (m)')
    plt.title('Mad River longitudinal profile and knickpoints > 3m')
    plt.plot(x, y)
    for index, val in enumerate(marker_x):
        plt.plot(marker_x[index], marker_y[index], 'r.', label='Knickpoint')
    plt.savefig('data_outputs/longitudinal_profile.jpeg')
    pd.DataFrame({'cross-section_id':transect_knickpoints}).to_csv('data_outputs/potential_knickpoints.csv', index=False)


def plot_longitudinal_bf(reach_name):
    bankfull_topo_detrend = pd.read_csv('data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    bankfull_topo = pd.read_csv('data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    bankfull_benchmark_detrend = pd.read_csv('data_outputs/bankfull_benchmark_detrend.csv')
    bankfull_benchmark = pd.read_csv('data_outputs/bankfull_benchmark.csv')
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    
    # Calc bankfull ranges for plotting
    benchmark_25_detrend = np.nanpercentile(bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], 25)
    benchmark_75_detrend = np.nanpercentile(bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], 75)
    topo_25_detrend = np.nanpercentile(bankfull_topo_detrend['bankfull'], 25)
    topo_75_detrend = np.nanpercentile(bankfull_topo_detrend['bankfull'], 75)

    # Extract and detrend thalweg for plotting
    thalwegs = all_widths_df['thalweg_elev']
    thalwegs_detrend = []
    x_vals_thalweg = np.arange(0, len(thalwegs))
    x = np.array(x_vals_thalweg).reshape(-1, 1)
    y = np.array(thalwegs)
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from thalwegs
    for index, val in enumerate(thalwegs):
        thalwegs_detrend.append(val - fit_slope[index])
    breakpoint()

    # Plot bankfull results along logitudinal profile for detrended data
    transect_spacing = 1 # units count - can figure out meters later
    x_len = len(bankfull_topo_detrend)
    x_vals = np.arange(0, (x_len * transect_spacing), transect_spacing)
    fig, ax = plt.subplots()
    plt.xlabel('Transects from upstream to downstream (numbered)')
    plt.ylabel('Bankfull elevation ASL (m)')
    plt.title('Logitudinal profile of bankfull elevations, Mad River (detrended)')
    plt.plot(x_vals, bankfull_topo_detrend['bankfull'], label='Topographic bankfull')
    plt.plot(x_vals, bankfull_benchmark_detrend['benchmark_bankfull_ams_detrend'], color='green', label='Benchmark bankfull')
    plt.plot(x_vals, thalwegs_detrend, color='grey', label='Thalweg (detrended)')
    plt.axhline(benchmark_25_detrend, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axhline(benchmark_75_detrend, linestyle='dashed', color='black') 
    plt.axhline(topo_25_detrend, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.axhline(topo_75_detrend, linestyle='dashed', color='grey')
    plt.legend(loc='upper right')
    plt.savefig('data_outputs/{}/Bankfull_longitudinals_detrended'.format(reach_name))
    plt.close()

    # Plot bankfull results along logitudinal profile for non-detrended data
    # Calc bankfull ranges for plotting
    benchmark_25 = np.nanpercentile(bankfull_benchmark['benchmark_bankfull_ams'], 25)
    benchmark_75 = np.nanpercentile(bankfull_benchmark['benchmark_bankfull_ams'], 75)
    topo_25 = np.nanpercentile(bankfull_topo['bankfull'], 25)
    topo_75 = np.nanpercentile(bankfull_topo['bankfull'], 75)

    transect_spacing = 1 # units count - can figure out meters later
    x_len = len(bankfull_topo)
    x_vals = np.arange(0, (x_len * transect_spacing), transect_spacing)
    fig, ax = plt.subplots()
    plt.xlabel('Transects from upstream to downstream (numbered)')
    plt.ylabel('Bankfull elevation ASL (m)')
    plt.title('Logitudinal profile of bankfull elevations, Mad River')
    plt.plot(x_vals, bankfull_topo['bankfull'], label='Topographic bankfull')
    plt.plot(x_vals, bankfull_benchmark['benchmark_bankfull_ams'], color='green', label='Benchmark bankfull')
    plt.plot(x_vals, thalwegs, color='grey', label='Thalweg')
    plt.axhline(benchmark_25, linestyle='dashed', color='black', label='Benchmark bankfull 25%-75%') 
    plt.axhline(benchmark_75, linestyle='dashed', color='black') 
    plt.axhline(topo_25, linestyle='dashed', color='grey', label='Topographic bankfull 25%-75%')
    plt.axhline(topo_75, linestyle='dashed', color='grey')
    plt.legend(loc='upper right')
    plt.savefig('data_outputs/{}/Bankfull_longitudinals'.format(reach_name))
    plt.close()

def plot_bankfull_increments(d_interval, reach_name):
    # bankfull_topo = pd.read_csv('data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    # bankfull_benchmark = pd.read_csv('data_outputs/{}/bankfull_benchmark.csv'.format(reach_name))
    agg_bankfull = pd.read_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'widths'] = eval(row['widths'])

    # First plot non-detrended widths. Add topo bankfull on each line. 
    # pull up the cross-section
    # pull up the topo bankfull elevation
    # find elevation on w/elev that matches most closely with topo bankfull. mark that point
    # plot marked point as a red dot on cross-section line. 

    # Detrend widths before plotting based on thalweg elevation, and start plotting point based on detrend
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope] # unnest the array
    
    # Create color ramp 
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=len(all_widths_df)-1)
    # Plot all widths spaghetti style
    fig, ax = plt.subplots()
    plt.ylabel('Channel width (m)')
    plt.xlabel('Detrended elevation (m)')
    plt.title('Incremental channel widths for Mad River')
    for index, row in all_widths_df.iterrows():
        row = row['widths']
        x_len = round(len(row) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        # apply detrend shift to xvals
        x_vals = [x_val - fit_slope[index] - intercept for x_val in x_vals]
        window_size = 3
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(row, weights, mode='same') # apply smoothing to smooth over artifacts in width series
        plt.plot(x_vals, smoothed, alpha=0.5, color=cmap(norm(index)), linewidth=2) 
    # plt.axvline(bankfull_width, label='Median width at modeled bankfull'.format(str(median_bankfull)), color='black', linewidth=0.75)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set array to avoid warnings
    cbar = plt.colorbar(sm, ax=ax)
    plt.xlim(-2.5, 10)
    # plt.ylim(0, 100)
    cbar.set_label("Downstream distance (m)")
    plt.savefig('data_outputs/{}/all_widths_detrend.jpeg'.format(reach_name), dpi=400)
    plt.close()

    # Plot average and bounds on all widths
    # calc element-wise avg, 25th, & 75th percentile of each width increment
    # bankfull_benchmark = bankfull_benchmark['benchmark_bankfull_ams']
    # bankfull_topo = bankfull_topo['bankfull']

    # non- detrended method for aggregation
    all_widths_df['x_vals'] = all_widths_df['widths'] 
    for index, row in all_widths_df.iterrows():
        row = row['widths']
        x_len = round(len(row) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        x_vals = [x_val - fit_slope[index] - intercept + 50 for x_val in x_vals]
        # Save detrended widths
        pd.DataFrame({'':x_vals, 'widths':row}).to_csv('data_outputs/{}/all_widths_detrended/widths_{}.csv'.format(reach_name, index), index=False) # Save widths
        all_widths_df.at[index, 'x_vals'] = str(x_vals)

    max_len = max(all_widths_df['widths'].apply(len)) # find the longest row in df
    max_len_xvals = max(all_widths_df['x_vals'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded'] = all_widths_df['widths'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    all_widths_df['x_vals_padded'] = all_widths_df['x_vals'].apply(lambda x: np.pad(x, (0, max_len_xvals - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    
    # Convert padded_xvals back to floats
    for index, row in all_widths_df.iterrows():
        all_widths_df.at[index, 'x_vals_padded'] = eval(row['x_vals_padded'])
    
    padded_df = pd.DataFrame(all_widths_df['widths_padded'].tolist())
    
    padded_df = padded_df.apply(lambda lst: [float(i) for i in lst])  # or float(i)
    transect_50 = padded_df.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25 = padded_df.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75 = padded_df.apply(lambda row: np.nanpercentile(row, 75), axis=0)
    padded_xvals = pd.DataFrame(all_widths_df['x_vals_padded'].tolist())
    transect_50_xvals = padded_xvals.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25_xvals = padded_xvals.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75_xvals = padded_xvals.apply(lambda row: np.nanpercentile(row, 75), axis=0)

    # Detrended, aggregated cross-sections using padded-zeros approach
    all_widths_df['widths_detrend'] = [[] for _ in range(len(all_widths_df))] 
    # Loop through all_widths
    for index, row in all_widths_df.iterrows():
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if offset_int < 0: # most likely case, downstream xsections are lower elevation than furthest upstream
            # populate new column of df with width values
            all_widths_df.loc[index, 'widths_detrend'].extend([0] * abs(offset_int) + row['widths']) # add zeros to beginning of widths list. Need to unnest when using.
        elif offset_int > 0: # this probably won't come up
            all_widths_df.loc[index, 'widths_detrend'].extend(row[abs(offset_int):])
        else:
            all_widths_df.loc[index, 'widths_detrend'].extend(row['widths'])
    
    # Once all offsets applied, use zero-padding aggregation method just like with non-detrended widths.
    max_len = max(all_widths_df['widths_detrend'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded_detrend'] = all_widths_df['widths_detrend'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df_detrend = pd.DataFrame(all_widths_df['widths_padded_detrend'].tolist())
    transect_50_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    transect_25_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 25), axis=0)
    transect_75_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 75), axis=0)

    # Determine x-vals for plotting
    x_range = range(0, len(transect_50_detrend))
    x_vals = list(x_range)
    x_vals = [i * d_interval for i in x_vals]
    # Determine x left-limit for plotting for each figure
    left_lim = 0
    while transect_25_detrend[left_lim] == 0:
        left_lim += 1
    left_lim = left_lim * d_interval - 5
    plt.plot(x_vals, transect_50_detrend, color='black', label='width/elevation median')
    plt.plot(x_vals, transect_25_detrend, color='blue', label='width/elevation 25/75th percentile')
    plt.plot(x_vals, transect_75_detrend, color='blue')
    for index, x in enumerate(agg_bankfull['pos_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='red', label='positive inflections', alpha=0.5)
        else:
            plt.axvline(x*d_interval, color='red', alpha=0.5)
    for index, x in enumerate(agg_bankfull['neg_inflections']):
        if index == 0:
            plt.axvline(x*d_interval, color='blue', label='negative inflections', alpha=0.5)
        else:
            plt.axvline(x*d_interval, color='blue', alpha=0.5)
    plt.xlabel('Elevation (m)')
    plt.ylabel('Channel width (m)')
    if reach_name == 'upper':
        plt.legend()
    plt.xlim(left = left_lim)
    # plt.ylim(0, 60)
    plt.legend()
    plt.savefig('data_outputs/{}/width_elev_aggregate_detrend.jpeg'.format(reach_name), dpi=500)
    # Save plot lines for later plotting
    plot_df = pd.DataFrame({'50th':transect_50_detrend, '25th':transect_25_detrend, '75th':transect_75_detrend})
    plot_df.to_csv('data_outputs/{}/width_elev_plotlines.csv'.format(reach_name))
    plt.close()

def transect_plot(transects, dem, plot_interval, d_interval, bankfull_boundary, reach_name):
    topo_bankfull = pd.read_csv('data_outputs/{}/bankfull_topo.csv'.format(reach_name))
    inflections = pd.read_csv('data_outputs/{}/max_inflections_aggregate.csv'.format(reach_name))
    all_widths_df = pd.read_csv('data_outputs/{}/all_widths.csv'.format(reach_name))

    # Use thalweg to detrend elevation on y-axes for transect plotting. Don't remove intercept (keep at elevation) 
    x = np.array(all_widths_df['transect_id']).reshape((-1, 1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    
    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        intersect_pts = line.geometry.intersection(bankfull_boundary)
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        # Add detrend value to elevs
        elevs = [i - fit_slope[transects_index] for i in elevs]

        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot = [bankfull_z[0][0], bankfull_z[1][0]] # elevation of benchmark bankfull for plotting
        bankfull_z_plot = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies

        # bring in topo bankfull for plotting
        current_topo_bankfull = topo_bankfull['bankfull'][transects_index]

        # Arrange points together for plotting
        def get_x_vals(y_vals):
            x_len = round(len(y_vals) * d_interval, 4)
            x_vals = np.arange(0, x_len, d_interval)
            return(x_vals)
        min_y = min(elevs)
        fig = plt.figure(figsize=(8,8))
        plt.plot(distances, elevs, color='black', linestyle='-', label='Cross section')
        # plt.axhline(current_topo_bankfull, color='grey', linestyle='dashed', label='Topographic-derived bankfull')
        # plt.axhline(bankfull_z_plot, color='red', linestyle='-', label='Benchmark bankfull')
        for index, x in enumerate(inflections['pos_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='red', label='positive inflections', alpha=0.5)
            else:
                plt.axhline(x*d_interval, color='red', alpha=0.5)
        for index, x in enumerate(inflections['neg_inflections']):
            if index == 0:
                plt.axhline(x*d_interval, color='blue', label='negative inflections', alpha=0.5)
            else:
                plt.axhline(x*d_interval, color='blue', alpha=0.5)
        plt.xlabel('Cross section distance (meters)', fontsize=12)
        plt.ylabel('Elevation (meters)', fontsize=12)
        plt.legend(fontsize=12)
        # increase font size for axes and labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Make the bottom of the ylim fall a meter below the lowest point in cross section
        plt.tight_layout()
        plt.savefig('data_outputs/{}/transect_plots_aggregate/bankfull_transect_{}.jpeg'.format(reach_name, transects_index))
        plt.close()

def stacked_width_plots(d_interval):
    # Bring in all plot lines from upper, mid, lower
    upper_plotlines = pd.read_csv('data_outputs/upper/width_elev_plotlines.csv')
    mid_plotlines = pd.read_csv('data_outputs/middle/width_elev_plotlines.csv')
    lower_plotlines = pd.read_csv('data_outputs/lower/width_elev_plotlines.csv')
    # Determine where to begin plotting based on where median line goes above zero
    def start_plot(line_25th):
        for index, value in line_25th.items():
            if value > 0:
                return index
    upper_plot_start = start_plot(upper_plotlines['25th'])
    upper_plotlines = upper_plotlines[upper_plot_start:].reset_index()
    middle_plot_start = start_plot(mid_plotlines['25th'])
    mid_plotlines = mid_plotlines[middle_plot_start:].reset_index()
    lower_plot_start = start_plot(lower_plotlines['25th'])
    lower_plotlines = lower_plotlines[lower_plot_start:].reset_index()

    def get_x_vals(y_vals):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
    x_upper = get_x_vals(upper_plotlines['50th'][:110])
    x_mid = get_x_vals(mid_plotlines['50th'][:110])
    x_lower = get_x_vals(lower_plotlines['50th'][:110])
    
    fig = plt.figure(figsize=(6,4))
    plt.plot(x_upper, upper_plotlines['50th'][:110], color='orange', label='upper reach')
    plt.plot(x_mid, mid_plotlines['50th'][:110], color='green',label='middle reach')
    plt.plot(x_lower, lower_plotlines['50th'][:110], color='blue',label='lower reach')
    plt.xlabel('Relative elevation (m)')
    plt.ylabel('Channel width (m)')
    plt.legend()

    breakpoint()
    plt.savefig('data_outputs/stacked_width_elev.jpeg')
    # Plot each median line based on starting point, with a distinctive color for each. 

    return