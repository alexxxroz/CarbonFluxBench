import yaml
import numpy as np
import pandas as pd
import xarray as xr

import os
from os import listdir
from os.path import join

config_fname = '../config.yaml'
with open(config_fname, 'r') as file:
    config = yaml.safe_load(file)

targets = ['NPP', 'Rh', 'NEE']
df = pd.read_parquet('../data/target_fluxes.parquet')
df['TIMESTAMP'] = pd.to_datetime(df.TIMESTAMP, format='%Y%m%d')

path=config['micasa_path']
files = sorted([x for x in listdir(path) if 'daily' in x])

if os.path.exists('../data/MiCASA.parquet'):
    data = pd.read_parquet('../data/MiCASA.parquet')
    d = data.to_dict(orient='list')
    files = [x for x in files if pd.to_datetime(x.split('.')[0].split('_')[-1], format='%Y%m%d')>pd.to_datetime(d['TIMESTAMP'][-1])]
else:
    d = {x: [] for x in ['TIMESTAMP', 'site'] + targets}
    sites = []
t = 0
for f_name in files:
    f_date = f_name.split('.')[0].split('_')[-1]
    try:
        ds = xr.open_dataset(join(path, f_name))
        
        date = ds.time.values[0]

        for idx, group in df.groupby(['site', 'lat', 'lon']):
            site, lat, lon = idx[0], float(idx[1]), float(idx[2])
            for target in targets:
                flux = ds.sel(lat=lat, lon=lon, method='nearest')[target].values[0]*1e3*60*60*24 #kg m-2 s-1 -> g m-2 d-1
                if np.isnan(flux):
                    print(f'Skipping site {site} {target} -- no pixel was found...')
                d[target].append(flux)
            d['site'].append(site)
            d['TIMESTAMP'].append(date)

        print(f'{f_name} is processed')
    except Exception as e:
        print(f'{e}\tSkipped {f_name}')
        continue
    final_df = pd.DataFrame(d)
    final_df.drop_duplicates(inplace=True)
    final_df.to_parquet('../data/MiCASA.parquet', index=None)
print('Done!')