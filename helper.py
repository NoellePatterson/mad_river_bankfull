# Quick helper file to combine two plots into one
import glob 
import matplotlib.pyplot as plt

reach_name = 'lower'

agg_plots = glob.glob('data_outputs/{}/transect_plots_aggregate/*'.format(reach_name))
xs_plots = glob.glob('data_outputs/{}/transect_plots/*'.format(reach_name))

# put files in numerical order based on number in filename
agg_plots.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
xs_plots.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# combine plots into one figure
# fig, axs = plt.subplots(len(agg_plots), 2, figsize=(12, len(agg_plots) * 6))

for i, (agg_plot, xs_plot) in enumerate(zip(agg_plots, xs_plots)):
    # fix this plotting code not working
    fig, ax = plt.subplots(1, 2)
    agg_img = plt.imread(agg_plot)
    xs_img = plt.imread(xs_plot)

    ax[0].imshow(agg_img)
    ax[0].set_title('Aggregate Plot {}'.format(i + 1))
    ax[0].axis('off')

    ax[1].imshow(xs_img)
    ax[1].set_title('Cross Section Plot {}'.format(i + 1))
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig('data_outputs/{}/cross_section_comparison/xs_{}'.format(reach_name, i))
    plt.close()

breakpoint()

# Quick one-off for counting ffc reference water years
import pandas as pd
import numpy as np
import glob

gage_metrics_file = glob.glob('data_inputs/35_GSLB_ref_gages/*/*annual_flow_result.csv')

wy_counter = 0
gage_metrics = []
for file in gage_metrics_file:
    data = pd.read_csv(file)
    dat_cols = data.columns[1:]
    for col in dat_cols:
        data_col = data[col]
        nan_counter = 0
        dat_len = len(data_col)
        for metric in data_col:
            if np.isnan(metric) == True:
                nan_counter += 1
        if nan_counter == dat_len:
            continue
        else:
            wy_counter += 1
print('number of reference water years = {}'.format(wy_counter))
breakpoint()