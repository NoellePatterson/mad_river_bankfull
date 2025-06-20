'''
NWS Query API
code modified from CIROH
'''
import io
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import requests

import dataretrieval.nwis as nwis

# Steps for implementation
# 1. Obtain API Key from CIROH to get access to BigQuery data free of charge. Email Arpita.
# 2. Set endpoint to access the correct retrospective data, according to this webpage: 
#  https://nwm-api.ciroh.org/docs#/default/flood_return_periods_return_period_get
# 3. Set correct reach ID's for analysis. I've found they align with comid's from the NHDplusV2 dataset. 
# Of the zip files available select the 'NHD_snapshot' one
# Flowlines with comid can be downloaded for any region in the US and tracked down in QGIS:

reach_name = 'lower' # choose either: 'upper', 'middle', or 'lower'

# define the reach feature id to run the queries for
if reach_name == 'lower':
    reach_id = 4577590
elif reach_name == 'middle':
    reach_id = 4577668
elif reach_name == 'upper':
    reach_id = 4577784
     

API_KEY = 'AIzaSyA7Y8s8ZV1mDm4YMJuSqCyxbZqtaCgzXEU'
API_URL = 'https://nwm-api.ciroh.org'
RETURN_PERIOD_ENDPOINT = f'{API_URL}/return-period'

header = {
    'x-api-key': API_KEY
}

params = {
    'comids': reach_id,
    'output_format': 'csv',
}
     

def get_return_period():
    r = requests.get(RETURN_PERIOD_ENDPOINT, params=params, headers=header)
    if r.status_code == 200:
        df = pd.read_csv(io.StringIO(r.text))
    else:
        raise requests.exceptions.HTTPError(r.text)
    return df
     

df = get_return_period()
df.to_csv('data_outputs/{}/flood_frequency.csv'.format(reach_name))
# Data appear to be in cms, but I didn't expect that

