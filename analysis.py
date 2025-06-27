import pandas as pd
import numpy as np
import math
import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import linregress
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks
import seaborn as sns
import re

def get_x_vals(y_vals, d_interval):
        x_len = round(len(y_vals) * d_interval, 4)
        x_vals = np.arange(0, x_len, d_interval)
        return(x_vals)
    
def multipoint_slope(windowsize, timeseries, xvals):
    dw = np.zeros(len(timeseries))
    lr_window = int(windowsize/2) # indexing later requires this to be an integer
    for n in range(lr_window, len(timeseries) - lr_window):
        x = xvals[n - lr_window:n + lr_window]
        y = timeseries[n - lr_window:n + lr_window]
        # remove nans with a mask, if there are at least two real data points
        nancount = sum(1 for x in y if isinstance(x, float) and math.isnan(x))
        if nancount < 3:
            mask = ~np.isnan(x) & ~np.isnan(y)
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x[mask], np.array(y)[mask])
        else: 
            slope1, intercept1, r_value1, p_value1, std_err1 = linregress(x, np.array(y))
        dw[n] = slope1  
    return dw  
    
def find_boundary(xsection, bound):
    for index, val in enumerate(xsection):
        if val > bound: # find first instance of exceeding lower width bound
            bound_index = index
            break
    # if upper bound not exceeded, set to last width index of cross section
    bound_index = index 
    return bound_index

def id_benchmark_bankfull(transects, dem, d_interval, bankfull_boundary, plot_interval):
    # For each transect, find intersection points with bankfull, and plot transects with intersections
    bankfull_benchmark = []
    thalweg_distances = []
    for index, row in transects.iterrows():
        line = gpd.GeoDataFrame({'geometry': [row['geometry']]}, crs=transects.crs)
        intersect_pts = line.geometry.intersection(bankfull_boundary)

        # Generate a spaced interval of stations along each transect for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))
        
        # Get Y and Z coordinates for bankfull intersections 
        station_zero = stations['geometry'][0]
        station_zero_gpf = gpd.GeoDataFrame({'geometry':[station_zero]}, crs=transects.crs)
        station_zero_gpf = station_zero_gpf.set_geometry('geometry') # ensure geometry is set correctly
        
        if isinstance(intersect_pts.geometry[0], Point):
            print('Only one intersection point identified; skipping to next cross section.')
            bankfull_benchmark.append(np.nan)
            continue
        coords = [(point.x, point.y) for point in intersect_pts.geometry[0].geoms]
        bankfull_z = list(dem.sample(coords))
        bankfull_z_plot_avg = np.nanmean([bankfull_z[0][0], bankfull_z[1][0]]) # Use average value of bankfull to smooth out inconsistencies
        bankfull_benchmark.append(bankfull_z_plot_avg)

        thalweg = min(elevs)
        # find station coordinates at thalweg - to get transect distances for detrending
        if index > 0: # measure distances for all but first (most upstream) transect
            thalweg_index = elevs.index(thalweg)
            thalweg_coords = stations.geometry[thalweg_index]
            # Get distance from thalweg to next thalweg
            next_transect = transects.iloc[index - 1]
            next_line = gpd.GeoDataFrame({'geometry': [next_transect['geometry']]}, crs=transects.crs)
            next_tot_len = next_line.length
            next_distances = np.arange(0, next_tot_len[0], plot_interval) 
            next_stations = next_transect['geometry'].interpolate(next_distances) # specify stations in transect based on plotting interval
            next_stations = gpd.GeoDataFrame(geometry=next_stations, crs=transects.crs)
            next_elevs = list(dem.sample([(point.x, point.y) for point in next_stations.geometry]))
            next_thalweg = min(next_elevs)
            next_thalweg_index = next_elevs.index(next_thalweg)
            next_thalweg_coords = next_stations.geometry[next_thalweg_index]
            thalweg_distance = next_thalweg_coords.distance(thalweg_coords) # distance to next thalweg, in meters
        else: 
            thalweg_distance = 0
        thalweg_distances.append(thalweg_distance)

    # Detrend benchmark bankfull results
    x_vals = thalweg_distances
    x = np.array(x_vals).reshape(-1, 1) 
    y = np.array(bankfull_benchmark).reshape(-1, 1)
    # drop nan values from bankfull results
    y = y[~np.isnan(y)]
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope =  slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from bankfull results
    bankfull_benchmark_detrend = []
    for index, val in enumerate(bankfull_benchmark):
        bankfull_benchmark_detrend.append(val - fit_slope[index]) 
    benchmark_bankfull_df = pd.DataFrame({'benchmark_bankfull_ams':bankfull_benchmark})
    benchmark_bankfull_df.to_csv('data_outputs/bankfull_benchmark.csv')
    benchmark_bankfull_detrend_df = pd.DataFrame({'benchmark_bankfull_ams_detrend':bankfull_benchmark_detrend})
    benchmark_bankfull_detrend_df.to_csv('data_outputs/bankfull_benchmark_detrend.csv')
    return()

def calc_dwdh(transects, dem, plot_interval, d_interval, width_calc_method, reach_name):
    # Loop through xsections and create dw/dh array for each xsection
    # df to store arrays of w and h
    all_widths_df = pd.DataFrame(columns=['widths'])
    incomplete_intersection_counter = 0
    total_measurements = 0

    # Optionally: check width at benchmark bankfull for each transect, store in a separate list
    bankfull_width_ls = [] # using benchmark bankfull, track channel width at bankfull for each transect
    benchmark_bankfull_df = pd.read_csv('data_outputs/bankfull_benchmark.csv')
    median_benchmark_bankfull = np.nanmedian(benchmark_bankfull_df['benchmark_bankfull_ams'])

    # for transect in transects:
    for transects_index, transects_row in transects.iterrows():
        wh_ls = []
        line = gpd.GeoDataFrame({'geometry': [transects_row['geometry']]}, crs=transects.crs) 
        # Generate a spaced interval of stations along each xsection for plotting
        tot_len = line.length
        distances = np.arange(0, tot_len[0], plot_interval) 
        stations = transects_row['geometry'].interpolate(distances) # specify stations in transect based on plotting interval
        stations = gpd.GeoDataFrame(geometry=stations, crs=transects.crs)
        # Extract z elevation at each station along transect
        elevs = list(dem.sample([(point.x, point.y) for point in stations.geometry]))

        # Determine total depth of iterations based on max rise on the lower bank
        min_z = min(elevs)
        min_y = list(elevs).index(min_z)
        max_left_bank = max(list(elevs)[0:min_y])
        max_right_bank = max(list(elevs)[min_y:])

        if max_right_bank < max_left_bank:
            max_depth = max_right_bank
        else:
            max_depth = max_left_bank

        # Shorten cross-sections if necessary to only include rising banks. Will not affect max depth. 
        while elevs[0] < max_left_bank:
            elevs = elevs[1:] # remove left-most point if it is below maximum left bank elevation (banks drop off)
        while elevs[-1] < max_right_bank:# remove right-most point if it is below maximum right bank elevation (banks drop off)
            elevs = elevs[:-1]
        
        depths = max_depth//d_interval
        # If depth is improperly assigned skip to next transect
        if depths[0] == float('inf'):
            continue
        
        # calc width at the current depth 
        if width_calc_method == 'partial':
            for index, depth in enumerate(range(int(depths[0]))):
                total_measurements += 1
                # find intercepts of current d with bed profile (as locs where normalized profile pts have a sign change)
                wat_level = [x - (d_interval * index) for x in elevs]
                intercepts = []
                for i, val in enumerate(elevs[0:-1]):
                    if np.sign(wat_level[i] * wat_level[i + 1]) < 0:
                        intercepts.append(distances[i])
                
                # Find distances between intercept points
                if len(intercepts) == 2: # most common case, one intercept on each side of channel
                    width = intercepts[1] - intercepts[0]
                elif (len(intercepts) % 2) == 0: # other common case, bed elevation has at least one extra pair of intercepts
                    partial_widths = []
                    for int_index in range(0, len(intercepts), 2):
                        w = intercepts[int_index + 1] - intercepts[int_index]
                        partial_widths.append(w)
                    width = sum(partial_widths)
                elif len(intercepts) == 3:
                    left_bank = min(intercepts)
                    right_bank = max(intercepts)
                    width = right_bank - left_bank
                else:
                    # print("Cannot accurately determine width with incomplete xsection intersection. Num intersections = {}.".format(len(intercepts)))
                    width = np.nan
                    incomplete_intersection_counter += 1 
                wh_ls.append(width)

                depth_0elev = round(d_interval * depth, 1)
                bm_bankfull = round(median_benchmark_bankfull, 1)
                if depth_0elev == bm_bankfull:
                    bankfull_width_ls.append(width)

        elif width_calc_method == 'continuous':
            thalweg = min(elevs) 
            thalweg_index = elevs.index(thalweg)
            for index, depth in enumerate(range(int(depths[0]))):
                total_measurements += 1
                # find intercepts of current d with bed profile (as locs where normalized profile pts have a sign change)
                wat_level = [x - (d_interval * index) for x in elevs]
                intercepts = []
                for i, val in enumerate(elevs[0:-1]):
                    if np.sign(wat_level[i] * wat_level[i + 1]) < 0:
                        intercepts.append(distances[i])
                
                # Find distances between intercept points
                width = np.nan
                if len(intercepts) == 0: 
                    width = 0
                elif len(intercepts) == 2: # most common case, one intercept on each side of channel
                    width = intercepts[1] - intercepts[0]
                elif (len(intercepts) % 2) == 0: # other common case, bed elevation has at least one extra pair of intercepts
                    # identify which pair of intercepts contains thalweg. select that pair for width calc. 
                    for int_index in range(0, len(intercepts), 2): # loop through each pair of intercepts
                        if intercepts[int_index] <= thalweg_index <= intercepts[int_index + 1]:
                            # found the intercepts pair that contains the thalweg index
                            width = intercepts[int_index + 1] - intercepts[int_index]
                            break
                elif len(intercepts) == 3: # This is assumed to mean that left and right banks are separated by a single point
                    left_bank = min(intercepts)
                    right_bank = max(intercepts)
                    width = right_bank - left_bank
                else:
                    print("Cannot accurately determine width with incomplete xsection intersection at cross-section {}. Num intersections = {}.".format(transects_index, len(intercepts)))
                    width = np.nan
                    incomplete_intersection_counter += 1 
                wh_ls.append(width)

                depth_0elev = round(d_interval * depth, 1)
                bm_bankfull = round(median_benchmark_bankfull, 1)
                if depth_0elev == bm_bankfull:
                    bankfull_width_ls.append(width)
        
        thalweg = min(elevs) # track this for use later in detrending
        # find station coordinates at thalweg
        if transects_index > 0: # measure distances for all but first (most upstream) transect
            thalweg_index = elevs.index(thalweg)
            thalweg_coords = stations.geometry[thalweg_index]
            # Get distance from thalweg to next thalweg
            next_transect = transects.iloc[transects_index - 1]
            next_line = gpd.GeoDataFrame({'geometry': [next_transect['geometry']]}, crs=transects.crs)
            next_tot_len = next_line.length
            next_distances = np.arange(0, next_tot_len[0], plot_interval) 
            next_stations = next_transect['geometry'].interpolate(next_distances) # specify stations in transect based on plotting interval
            next_stations = gpd.GeoDataFrame(geometry=next_stations, crs=transects.crs)
            next_elevs = list(dem.sample([(point.x, point.y) for point in next_stations.geometry]))
            next_thalweg = min(next_elevs)
            next_thalweg_index = next_elevs.index(next_thalweg)
            next_thalweg_coords = next_stations.geometry[next_thalweg_index]
            thalweg_distance = next_thalweg_coords.distance(thalweg_coords) # distance to next thalweg, in meters
        else: 
            thalweg_distance = 0
        
        wh_ls_df = pd.DataFrame({'widths':wh_ls})
        wh_ls_df.to_csv('data_outputs/{}/all_widths/widths_{}.csv'.format(reach_name, transects_index))
        wh_append = pd.DataFrame({'widths':[wh_ls], 'transect_id':transects_index, 'thalweg_elev':thalweg, 'thalweg_distance':thalweg_distance})
        all_widths_df = pd.concat([all_widths_df, wh_append], ignore_index=True)

    bankfull_width = np.nanmedian(bankfull_width_ls)
    all_widths_df.to_csv('data_outputs/{}/all_widths.csv'.format(reach_name))
    return(all_widths_df, bankfull_width)

def calc_derivatives(d_interval, all_widths_df, slope_window, lower_bound, upper_bound, width_calc_method, reach_name):
    # calculate and plot second derivative of width (height is constant)
    # Calc upper and lower bounds widths
    lower_ls = []
    upper_ls = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        for i, val in enumerate(xsection):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
        lower_height_index = start_width_index + lower_bound # height index at 0.5m above thalweg
        width_lower = xsection[start_width_index + lower_bound] # height index at 0.5m above thalweg
        lower_ls.append(width_lower)
        try:
            upper_height_index = start_width_index + upper_bound # height index at 10m above thalweg
            width_upper = xsection[start_width_index + upper_bound] # width at 10m above thalweg
            upper_ls.append(width_upper)
        except:
            continue
    lower = np.nanmedian(lower_ls)
    upper = np.nanmedian(upper_ls)

    topo_bankfull = []
    bankfull_width = []
    for x_index, xsection in enumerate(all_widths_df['widths']): # loop through all x-sections
        dw = []
        ddw = []
        xs_xvals = get_x_vals(xsection, d_interval)
        dw = multipoint_slope(slope_window, xsection, xs_xvals)
        ddw = multipoint_slope(slope_window, dw, xs_xvals)

        # Find max second derivative as bankfull. 
        ddw_abs = [abs(i) for i in ddw]
        # Add in upper and lower search bounds on max 2nd deriv bankfull ID
        for i, val in enumerate(xsection):
            if val > 0: # ID first instance when width exceeds zero, set as thalweg start point
                start_width_index = i
                break
        lower_height_index = start_width_index + lower_bound # height index at 0.5m above thalweg
        upper_height_index = start_width_index + upper_bound # height index at 10m above thalweg
        lower_bound_index = find_boundary(xsection, lower) # enter bound based on width
        upper_bound_index = find_boundary(xsection, upper) # enter bound based on width

        max_ddw = np.nanmax(ddw_abs[lower_height_index:upper_height_index]) # uncomment if using search limits 
        max_ddw_index = ddw_abs.index(max_ddw)
        bankfull_id_elevation = d_interval * max_ddw_index # sea-level elevation corresponding with bankfull
        topo_bankfull.append(bankfull_id_elevation)
        bankfull_width.append(xsection[max_ddw_index])

        # Output dw/ddw spacing, which is in 1/10 meter increments
        xsection_xvals = get_x_vals(xsection, d_interval)
        dw_xvals = get_x_vals(dw, d_interval)
        ddw_xvals = get_x_vals(ddw, d_interval)
        dw_df = pd.DataFrame({'elevation_m':dw_xvals, 'dw':dw})
        ddw_df = pd.DataFrame({'elevation_m':ddw_xvals, 'ddw':ddw})
        dw_df.to_csv('data_outputs/{}/first_order_roc/first_order_roc_{}.csv'.format(reach_name, x_index))
        ddw_df.to_csv('data_outputs/{}/second_order_roc/second_order_roc_{}.csv'.format(reach_name, x_index))

        # Plot derivative results for each cross section plus cross-section bankfull
        fig = plt.figure(figsize=(20,15))
        if reach_name == 'upper':
            left_lim = 270
        elif reach_name == 'middle':
            left_lim = 175
        elif reach_name == 'lower':
            left_lim = 0
        # xs_plot = mpimg.imread("data_outputs/{}/transect_plots/bankfull_transect_{}.jpeg".format(reach_name, x_index))
        # plt.subplot2grid((3,3), (0,2))
        # plt.plot(xsection_xvals, xsection)
        # plt.xlim(left = left_lim)
        # plt.axvline(max_ddw_index * d_interval, color='black')
        # plt.title('Incremental channel widths', fontsize=15)
        # plt.subplot2grid((3,3), (1,2))    
        # plt.plot(dw_xvals, dw)
        # plt.xlim(left = left_lim)
        # plt.axvline(max_ddw_index * d_interval, color='black')
        # plt.title('First derivative', fontsize=15)
        # plt.subplot2grid((3,3), (2,2))
        # plt.plot(ddw_xvals, ddw)
        # plt.xlim(left = left_lim)
        # plt.axvline(max_ddw_index * d_interval, label='2nd derivative abs maxima={}'.format(str(round(max_ddw_index * d_interval, 3)), color='black'))
        # plt.title('Second derivative', fontsize=15)
        # plt.legend(fontsize=14)
        # # Large plot spanning second column
        # plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=3)
        # plt.imshow(xs_plot)
        # fig.suptitle('Bankfull slope identification for Mad River with {}pt rolling avg, {} width method'.format(slope_window, width_calc_method), fontsize=20)
        # plt.savefig('data_outputs/{}/derivative_plots/{}.jpeg'.format(reach_name, x_index))
    
    # Use thalweg elevs to detrend bankfull elevation results. Don't remove intercept (keep at elevation) 
    # remove this if no longer needed. 
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    # pairwise subtract fit from bankfull results
    topo_bankfull_detrend = []
    for index, val in enumerate(topo_bankfull):
        topo_bankfull_detrend.append(val - fit_slope[index])

    # save topo-derived bankfull for each transect
    topo_bankfull_detrend_dt = pd.DataFrame({'bankfull':topo_bankfull_detrend})
    topo_bankfull_detrend_dt.to_csv('data_outputs/{}/bankfull_topo_detrend.csv'.format(reach_name))
    topo_bankfull_dt = pd.DataFrame({'bankfull':topo_bankfull})
    topo_bankfull_dt.to_csv('data_outputs/{}/bankfull_topo.csv'.format(reach_name))

    print('Done with topo bankfull identification')

    return(topo_bankfull, topo_bankfull_detrend)

def calc_derivatives_aggregate(d_interval, all_widths_df, slope_window, lower_bound, upper_bound, reach_name):

    # Function for identifying top inflection point peaks
    def top_peaks_id(peaks_array, num_peaks):
        if len(peaks_array[0]) < num_peaks:
            peak_range = len(peaks_array[0])
        else: 
            peak_range = num_peaks
        peak_indices = peaks_array[0]
        max_peaks = []
        for i in range(0, peak_range): # Here is where to define number of peaks looking for 
            current_max = 0 
            current_max_index = 0
            for j in range(len(peak_indices)):
                if abs(peaks_array[1]['peak_heights'][j]) > current_max:
                    current_max = abs(peaks_array[1]['peak_heights'][j])
                    current_max_index = j
            peaks_array[1]['peak_heights'] = np.delete(peaks_array[1]['peak_heights'], current_max_index)
            max_peaks.append(peak_indices[current_max_index])
            peak_indices = np.delete(peak_indices, current_max_index)
        return max_peaks

    # Use full array of 2nd deriv values to find range of inflection points across reach
    # Use thalweg elevs to detrend 2nd derivatives. Don't remove intercept (keep at elevation) 
    # create x variation which is an accumulated distance from the furthest upstream cross-section
    x = np.cumsum(all_widths_df['thalweg_distance'].values).reshape((-1,1))
    y = np.array(all_widths_df['thalweg_elev'])
    model = LinearRegression().fit(x, y)
    slope = model.coef_
    intercept = model.intercept_
    fit_slope = slope*x
    fit_slope = [val[0] for val in fit_slope]
    # breakpoint()
    '''
    Alt inflection point method: calc inflection points from aggregated width/depth curve
    '''
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
    n_xs = len(all_widths_df.index) # number of cross-sections to use when applying requirements for number of cross-sections in aggregation
    max_len = max(all_widths_df['widths_detrend'].apply(len)) # find the longest row in df
    all_widths_df['widths_padded_detrend'] = all_widths_df['widths_detrend'].apply(lambda x: np.pad(x, (0, max_len - len(x)), constant_values=np.nan)) # pad all shorter rows with nan
    padded_df_detrend = pd.DataFrame(all_widths_df['widths_padded_detrend'].tolist())
    # drop columns element-wise in which more than half of values are nan
    padded_df_detrend = padded_df_detrend.dropna(axis=1, thresh=n_xs * 0.5) # drop columns with less than 50% of values present
    # calculate transect_50_detrend as the median of each column
    # this is the aggregate cross-section for the reach
    transect_50_detrend = padded_df_detrend.apply(lambda row: np.nanpercentile(row, 50), axis=0)
    
    # take second derivative of transect_50_detrend
    xvals_agg = get_x_vals(transect_50_detrend, d_interval)
    dw = multipoint_slope(slope_window, transect_50_detrend, xvals_agg)
    ddw = multipoint_slope(slope_window, dw, xvals_agg)
    # use inflection pt method to find top pos/neg peaks
    inflections_array_agg = ddw
    peaks_pos_agg = find_peaks(inflections_array_agg, height=max(inflections_array_agg)/2, distance=5, width=2, prominence=20) # require peaks to be at least half the mag of max peak
    inflections_array_neg_agg = [-i for i in inflections_array_agg] # invert all signs to detect negative peaks
    peaks_neg_agg = find_peaks(inflections_array_neg_agg, height=max(inflections_array_neg_agg)/2, distance=5, width=2, prominence=20) # require peaks to be at least half the mag of max peak
    # ID top 3 peaks in each category - positive
    max_pos_peak_agg = top_peaks_id(peaks_pos_agg, 3)
    # ID top 3 peaks in each category - negative
    max_neg_peak_agg = top_peaks_id(peaks_neg_agg, 3)

    # Save values and plot results
    max_len_agg = max(len(peaks_pos_agg[0]), len(peaks_neg_agg[0]))
    pos_peak_indices_pad_agg = max_pos_peak_agg + [np.nan] * (max_len_agg - len(max_pos_peak_agg))
    neg_peak_indices_pad_agg = max_neg_peak_agg + [np.nan] * (max_len_agg - len(max_neg_peak_agg))
    max_inflections_df_agg = pd.DataFrame({'pos_inflections':pos_peak_indices_pad_agg, 'neg_inflections':neg_peak_indices_pad_agg})
    max_inflections_df_agg.to_csv('data_outputs/{}/max_inflections_aggregate.csv'.format(reach_name))

    # Determine x-vals for plotting
    x_range = range(0, len(inflections_array_agg))
    x_vals = list(x_range)
    x_vals = [i * d_interval - intercept for i in x_vals]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(x_vals, inflections_array_agg, color='black')
    plt.xlim(-5, 25)

    for index, peak in enumerate(max_pos_peak_agg):
        if index == 0:
            plt.axvline(peak/10 - intercept, color='red', label='positive inflections')
        else:
            plt.axvline(peak/10 - intercept, color='red')
    for index, peak in enumerate(max_neg_peak_agg):
        if index == 0:
            plt.axvline(peak/10 - intercept, color='blue', label='negative inflections')
        else:
            plt.axvline(peak/10 - intercept, color='blue')
    
    plt.title('Inflection Points Density, Aggregate Method')
    plt.xlabel('Detrended elevation (m)')
    plt.ylabel('Second derivative')
    # plt.text(.5, .5, str(round(max(inflections_array), 3)))
    plt.legend()
    plt.savefig('data_outputs/{}/inflection_pt_density_plot_agg.jpeg'.format(reach_name))
    plt.close()

    '''
    Inflection Point Methodology
    '''
    # Bring in all 2nd deriv arrays as a list
    inflections_fp = glob.glob('data_outputs/{}/second_order_roc/*.csv'.format(reach_name)) # not yet detrended
    # order the filepaths
    def extract_num(path):
        match = re.search(r'\d+', path)
        return int(match.group()) if match else np.nan 
    inflections_fp_sorted = sorted(inflections_fp, key=extract_num)

    inflections_ls = []
    for index, fp in enumerate(inflections_fp_sorted):
        inflection_df = pd.read_csv(fp)
        inflections = inflection_df['ddw'].tolist()
        # Incorporate detrend as a shift in 2nd derivative array
        # Should be raising all values after first transect. So they start later. 
        offset = fit_slope[index]
        offset = offset / d_interval
        offset_int = int(offset)
        if offset_int < 0:
            inflections = [0] * abs(offset_int) + inflections
        else: # Only other case is no detrend (first transect)
            inflections = inflections
        inflections_ls.append(inflections)
        
    # convert list of lists into dataframe
    inflections_df = pd.DataFrame(inflections_ls)
    n_xs = len(inflections_df.index)
    inflections_df = inflections_df.dropna(axis=1, thresh=n_xs * 0.5) # drop columns with less than 50% of values present
    # Aggregate all arrays together by averaging across all rows in df
    inflections_array = inflections_df.mean(axis=0, skipna=True)

    # identify top three peaks (across positive and negative)
    # prominence req needed for middle reach but doesn't work for upper reach. 
    peaks_pos = find_peaks(inflections_array, height=max(inflections_array)/2, distance=5, width=2) #, prominence=20) # require peaks to be at least half the mag of max peak
    inflections_array_neg = [-i for i in inflections_array] # invert all signs to detect negative peaks
    peaks_neg = find_peaks(inflections_array_neg, height=max(inflections_array_neg)/2, distance=5, width=2) #, prominence=20) # require peaks to be at least half the mag of max peak
    # save peak locs for plotting along wd and cross-sections
    # ID top 3 peaks in each category - positive
    max_pos_peak = top_peaks_id(peaks_pos, 3)

    # ID top 3 peaks in each category - negative
    max_neg_peak = top_peaks_id(peaks_neg, 3)

    # Plot results density-style
    # Determine x-vals for plotting
    x_range = range(0, len(inflections_array))
    x_vals = list(x_range)
    x_vals = [i * d_interval - intercept for i in x_vals]
    
    fig, ax = plt.subplots()
    plt.plot(x_vals, inflections_array, color='black')
    plt.xlim(-5, 25)

    for index, peak in enumerate(max_pos_peak):
        if index == 0:
            plt.axvline(peak/10 - intercept, color='red', label='positive inflections')
        else:
            plt.axvline(peak/10 - intercept, color='red')
    for index, peak in enumerate(max_neg_peak):
        if index == 0:
            plt.axvline(peak/10 - intercept, color='blue', label='negative inflections')
        else:
            plt.axvline(peak/10 - intercept, color='blue')
    
    plt.title('Inflection Points Density')
    plt.xlabel('Detrended elevation (m)')
    plt.ylabel('Second derivative')
    # plt.text(.5, .5, str(round(max(inflections_array), 3)))
    plt.legend()
    plt.savefig('data_outputs/{}/inflection_pt_density_plot_all.jpeg'.format(reach_name))
    plt.close()
    # Save max positive and negative inflections (bankfull range)
    max_len = max(len(max_pos_peak), len(max_neg_peak))
    pos_peak_indices_pad = max_pos_peak + [np.nan] * (max_len - len(max_pos_peak))
    neg_peak_indices_pad = max_neg_peak + [np.nan] * (max_len - len(max_neg_peak))
    max_inflections_df = pd.DataFrame({'pos_inflections':pos_peak_indices_pad, 'neg_inflections':neg_peak_indices_pad})
    max_inflections_df.to_csv('data_outputs/{}/max_inflections.csv'.format(reach_name))

    # Create a plot overlaying inflections_array_agg and inflections_array
    x_vals = get_x_vals(inflections_array, d_interval)
    x_vals_agg = get_x_vals(inflections_array_agg, d_interval)
    fig, ax = plt.subplots()
    plt.plot(x_vals_agg, inflections_array_agg, color='black', label='Aggregate Inflections')
    plt.plot(x_vals, inflections_array, color='green', label='Cross-section Inflections')
    plt.xlim(left=165)
    plt.xlabel('Detrended elevation (m)')
    plt.ylabel('Second derivative')
    plt.legend()
    plt.title('Inflection Points Density, Aggregate vs Cross-section')
    plt.savefig('data_outputs/{}/inflection_pt_density_plot_agg_vs_xs.jpeg'.format(reach_name))
    plt.close()
    breakpoint()
 # there it is! 