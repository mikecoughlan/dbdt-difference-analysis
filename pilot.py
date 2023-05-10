import gc
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

min_lat = 36
min_lon = 338
max_lat = 71
max_lon = 41

station_locations = pd.read_csv('supermag-stations-info.csv')

# fmt: off
stations_to_remove = ['A09', 'A10', 'A11', 'ALT', 'ARK', 'ASA', 'ASH', 'B01',
						'BJI', 'C09', 'CGO', 'CPL', 'CPY', 'CWE', 'DRB', 'E01',
						'E02', 'E03', 'E04', 'EUA', 'FTN', 'FVE', 'GLK', 'GRK',
						'KAU', 'KGD', 'KOR', 'KZN', 'LNN', 'M02', 'M11', 'MCE',
						'MHV', 'MNK', 'MSK', 'MZH', 'NAD', 'NKK', 'NR2', 'NVL',
						'PKS', 'PNL', 'POD', 'PRG', 'R01', 'R02', 'R03', 'R04',
						'R05', 'R06', 'R07', 'R08', 'R09', 'R10', 'RSV', 'SAH',
						'SAS', 'SKD', 'SMA', 'SUT', 'T26', 'T27', 'T60', 'T62',
						'TKT', 'TLK', 'TOL', 'TOR', 'TTB', 'TUL', 'W01', 'W02',
						'W03', 'W04', 'W05', 'WSE', 'WTK', 'YSS', 'KHS', 'BEY', 'KLI']
# fmt: on

station_locations = station_locations[
	station_locations['IAGA'].isin(stations_to_remove) == False
]

df = station_locations[
	(station_locations['GEOLAT'] >= min_lat)
	& (station_locations['GEOLAT'] <= max_lat)
	& (
		(station_locations['GEOLON'] >= min_lon)
		| (station_locations['GEOLON'] <= max_lon)
	)
]

df.reset_index(inplace=True, drop=True)

stations = df['IAGA'].tolist()

station_dict = {}
stations_df = pd.DataFrame()
for station in stations:
	temp_df = pd.read_feather(f'../data/supermag/{station}.feather')
	temp_df = temp_df[['Date_UTC', 'dbht']]
	temp_df.rename(columns={'dbht':station}, inplace=True)
	# temp_df.dropna(subset=['dbht'], inplace=True)
	temp_df.set_index('Date_UTC', drop=True, inplace=True)
	stations_df = pd.concat([stations_df, temp_df], axis=1, ignore_index=False)

	del temp_df
	gc.collect()

station_dict['stations_df'] = stations_df


main = []
for station in tqdm(stations):
	difference_df = pd.DataFrame()
	for stat in stations:
		if (station == stat) or (stat in main):
			continue
		difference_df[stat] = (
			station_dict['stations_df'][station] - station_dict['stations_df'][stat]
		).abs()

	difference_df.dropna(how='all', inplace=True)
	difference_df.reset_index(inplace=True, drop=False)
	difference_df.to_feather(f'outputs/{station}_differences.feather')
	# station_dict[station]['difference_df'] = difference_df
	# main.append(station)

	del difference_df
	gc.collect()

# with open('difference_dict.pkl', 'wb') as f:
# 	pickle.dump(station_dict, f)
