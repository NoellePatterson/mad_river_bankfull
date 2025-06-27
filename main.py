"""
Bankfull Intersection Identification

This script estimates the bankfull level within a river channel using topographic data. Applied here at Mad River, Vermont. 

Noelle Patterson, USU 
March 2025
"""

import os
import geopandas as gpd
import pandas as pd
import rasterio
import raster_footprint
import ast
from shapely.geometry import Point, shape, MultiPoint
from shapely.geometry.point import Point
import numpy as np
from matplotlib import pyplot as plt
from analysis import id_benchmark_bankfull, calc_dwdh, calc_derivatives, calc_derivatives_aggregate
from visualization import transect_plot, plot_bankfull_increments, plot_longitudinal_bf, plot_longitudinal_profile, stacked_width_plots
from spatial_analysis import trim_cross_section, create_bankfull_pts

# Steps for bankfull analysis:
# 1. Identify benchmark bankfull using inundation rasters (Analysis.py -> id_benchmark_bankfull)
# 2. Measure channel width along a depth interval for each cross-section (Analysis.py -> calc_dwdh)
# 3. Calculate first and second order derivatives of the channel widths to identify topographic bankfull (Analysis.py -> calc_derivatives)
# 4. Post-processing: plot results (Visualization.py -> plot_bankfull_increments, plot_longitudinal_bf)

reach_name = 'upper' # choose either: 'upper', 'middle', or 'lower'

# Input data file paths  
thalweg_fp = 'data_inputs/thalweg/Thalweg.shp'
bankfull_fp = 'data_inputs/wse/wse6.tif' 
dem_fp = 'data_inputs/dem/dem.tif'
if reach_name == 'upper':
    transect_fp = 'data_inputs/cross_sections/transects_upper_long.shp' 
elif reach_name == 'middle':
    transect_fp = 'data_inputs/cross_sections/transects_middle_long.shp' 
elif reach_name == 'lower':
    transect_fp = 'data_inputs/cross_sections/transects_lower_long.shp' 
else:
    # print message and exit
    print('Please choose a valid reach name: upper, middle, or lower')
    exit()

# Create output folders if needed
if not os.path.exists('data_outputs/{}/all_widths/'.format(reach_name)):
    os.makedirs('data_outputs/{}/all_widths/'.format(reach_name))
if not os.path.exists('data_outputs/{}/first_order_roc/'.format(reach_name)):
    os.makedirs('data_outputs/{}/first_order_roc/'.format(reach_name))
if not os.path.exists('data_outputs/{}/second_order_roc/'.format(reach_name)): 
    os.makedirs('data_outputs/{}/second_order_roc/'.format(reach_name))
if not os.path.exists('data_outputs/{}/transect_plots/'.format(reach_name)): 
    os.makedirs('data_outputs/{}/transect_plots/'.format(reach_name))
if not os.path.exists('data_outputs/{}/spatial/'.format(reach_name)): 
    os.makedirs('data_outputs/{}/spatial/'.format(reach_name))
if not os.path.exists('data_outputs/{}/all_widths_detrended'.format(reach_name)):
    os.makedirs('data_outputs/{}/all_widths_detrended'.format(reach_name))
if not os.path.exists('data_outputs/{}/derivative_plots'.format(reach_name)):
    os.makedirs('data_outputs/{}/derivative_plots'.format(reach_name))

# Upload input data: transects, stations, and bankfull raster 
transects = gpd.read_file(transect_fp)
thalweg = gpd.read_file(thalweg_fp)
bankfull = rasterio.open(bankfull_fp)
dem = rasterio.open(dem_fp)
# Convert bankfull raster into a footprint line object
bankfull_footprint = raster_footprint.footprint_from_rasterio_reader(bankfull, destination_crs = bankfull.crs)
bankfull_footprint = shape(bankfull_footprint)
bankfull_boundary = bankfull_footprint.boundary
bankfull_boundary = gpd.GeoDataFrame({'geometry':[bankfull_boundary]}, crs=bankfull.crs)

plot_interval = 1 # set plotting interval along transect in units of meters
d_interval = 10/100 # Set intervals to step up in depth (in units meters). 10cm intervals
slope_window = 10 # Set window size for calculating slope for derivatives
lower_bound = 5 # Set lower vertical boundary for bankfull id within cross-section, in units of d_interval. 5 = 50cm
upper_bound = 100 # Set upper vertical boundary for bankfull id within cross-section, in units of d_interval. 100 = 10m
spatial_plot_interval = 0.5 # interval for finding bankfull elevation along transects. in meters?
width_calc_method = 'partial' # 'continuous' 'partial' - choose from either partial additive widths or continuous-only methods of width calculation
start_index = 0 # If using cross-sections from the middle of a series optionally assign a starting index for plotting purposes


# output = id_benchmark_bankfull(transects, dem, d_interval, bankfull_boundary, plot_interval)
all_widths_df, bankfull_width = calc_dwdh(transects, dem, plot_interval, d_interval, width_calc_method, reach_name)
# print('Dwdh calc done!!')

# topo_bankfull, topo_bankfull_detrend = calc_derivatives(d_interval, all_widths_df, slope_window, lower_bound, upper_bound, width_calc_method, reach_name)
# print('Derivatives calc done!!')

output = calc_derivatives_aggregate(d_interval, all_widths_df, slope_window, lower_bound, upper_bound, reach_name)
# output = transect_plot(transects, dem, plot_interval, d_interval, bankfull_boundary, reach_name)

# breakpoint()
# print('Begin plotting')
# output = plot_longitudinal_bf(reach_name)
# output = stacked_width_plots(d_interval)
# output = plot_bankfull_increments(d_interval, reach_name)


# plot_longitudinal_profile()
# breakpoint()

# optional spatial analyses
# output = trim_cross_section(transects, thalweg, reach_name)
# output = create_bankfull_pts(transects, dem, thalweg, d_interval, spatial_plot_interval, reach_name)