import gc
import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

min_lat = 36
min_lon = 338
max_lat = 71
max_lon = 41

station_locations = pd.read_csv('supermag-stations-info.csv')
storm_list = pd.read_csv('stormList.csv', header=None, names=['dates'])		# loading the list of storms as defined by SYM-H minimum

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


def storm_extract(df, storm_list, lead, recovery):
	'''
		Pulling out storms using a defined list of datetime strings, adding a lead and recovery time to it and
		appending each storm to a list which will be later processed.

		Args:
			data (list of pd.dataframes): ACE and supermag data with the test set's already removed.
			storm_list (str or list of str): datetime list of storms minimums as strings.
			lead (int): how much time in hours to add to the beginning of the storm.
			recovery (int): how much recovery time in hours to add to the end of the storm.

		Returns:
			list: ace and supermag dataframes for storm times
			list: np.arrays of shape (n,2) containing a one hot encoded boolean target array
		'''
	storms = list()								# initalizing the lists
	temp_df = pd.DataFrame()

	# setting the datetime index
	df.reset_index(drop=False, inplace=True)
	pd.to_datetime(df['Date_UTC'], format='%Y-%m-%d %H:%M:%S')
	df.reset_index(drop=True, inplace=True)
	df.set_index('Date_UTC', inplace=True, drop=True)
	df.index = pd.to_datetime(df.index)

	stime, etime = [], []					# will store the resulting time stamps here then append them to the storm time df

	# will loop through the storm dates, create a datetime object for the lead and recovery time stamps and append those to different lists
	for date in storm_list:
		stime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))-pd.Timedelta(hours=lead))
		etime.append((datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))+pd.Timedelta(hours=recovery))

	# adds the time stamp lists to the storm_list dataframes
	storm_list['stime'] = stime
	storm_list['etime'] = etime
	for start, end in zip(storm_list['stime'], storm_list['etime']):		# looping through the storms to remove the data from the larger df
		storm = df[(df.index >= start) & (df.index <= end)]

		if len(storm) != 0:
			temp_df = pd.concat([temp_df, storm], axis=0, ignore_index=False)

	temp_df.dropna(how='all', inplace=True)

	return temp_df


def converting_from_degrees_to_km(lat_1, lon_1, lat_2, lon_2):

	mean_lat = (lat_1 + lat_2)/2
	x = lon_2 - lon_1
	y = lat_2 - lat_1
	dist_x = x*(111.320*math.cos(math.radians(mean_lat)))
	dist_y = y*110.574

	distance = math.sqrt((dist_x**2)+(dist_y**2))

	return distance

main = []
for station in tqdm(stations):

	if os.path.exists(f'outputs/{station}_storm_only_differences.feather'):
		continue

	difference_df = pd.DataFrame()
	for stat in stations:
		if (station == stat) or (stat in main):
			continue
		difference_df[stat] = (
			station_dict['stations_df'][station] - station_dict['stations_df'][stat]
		).abs()

	if difference_df.empty:
		continue
	difference_df = storm_extract(difference_df, storm_list['dates'], lead=12, recovery=24)		# extracting the storms using list method
	difference_df.reset_index(inplace=True, drop=False)
	difference_df.to_feather(f'outputs/{station}_storm_only_differences.feather')
	# station_dict[station]['difference_df'] = difference_df
	main.append(station)
	print(f'Difference DF columns (len {len(difference_df.columns)}): {difference_df.columns}')
	print(f'Main list: {main}')

	del difference_df
	gc.collect()

# with open('difference_dict.pkl', 'wb') as f:
# 	pickle.dump(station_dict, f)


fig = plt.figure(figsize=(20,15))
for station in stations:

	if os.path.exists(f'outputs/{station}_storm_only_differences.feather'):
		df = pd.read_feather(f'outputs/{station}_storm_only_differences.feather')
		df.set_index('Date_UTC', inplace=True, drop=True)

		lat1 = station_locations.loc[station_locations['IAGA']==station]['GEOLAT'].item()
		lon1 = station_locations.loc[station_locations['IAGA']==station]['GEOLON'].item()
		lon1 = (lon1 + 180) % 360 - 180

		for col in tqdm(df.columns):
			lat2 = station_locations.loc[station_locations['IAGA']==col]['GEOLAT'].item()
			lon2 = station_locations.loc[station_locations['IAGA']==col]['GEOLON'].item()
			lon2 = (lon2 + 180) % 360 - 180
			distance = converting_from_degrees_to_km(lat1, lon1, lat2, lon2)
			print(len(df[col]))
			y = df[col].dropna().tolist()
			print(len(y))
			x = [distance] * len(y)
			plt.scatter(x, y)
		gc.collect()

plt.savefig('difference_and_distance.png')
