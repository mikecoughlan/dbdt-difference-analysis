import gc
import math
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
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

# station_dict = {}
# stations_df = pd.DataFrame()
# for station in stations:
# 	temp_df = pd.read_feather(f'../data/supermag/{station}.feather')
# 	temp_df = temp_df[['Date_UTC', 'dbht']]
# 	temp_df.rename(columns={'dbht':station}, inplace=True)
# 	# temp_df.dropna(subset=['dbht'], inplace=True)
# 	temp_df.set_index('Date_UTC', drop=True, inplace=True)
# 	stations_df = pd.concat([stations_df, temp_df], axis=1, ignore_index=False)

# 	del temp_df
# 	gc.collect()

# station_dict['stations_df'] = stations_df


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

print(stations[:5])
print(stations[-5:])

fig = plt.figure(figsize=(25,18))
# ax2 = ax1.twinx()
x, extreme_values, station_0, station_1 = [], [], [], []
y_median, y_mean, y_topq, y_bottomq, y_std = [], [], [], [], []
i=0
for station in stations[:-1]:

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
			x.append(distance)
			y = df[col].dropna()
			y_median.append(y.median())
			y_mean.append(y.mean())
			y_topq.append(y.quantile(0.75))
			y_bottomq.append(y.quantile(0.25))
			y_std.append(y.std())
			extreme_values.append(y[y>y.quantile(0.9999)].tolist())
			station_0.append(station)
			station_1.append(col)

			if i==0:
				plt.Figure()
				plt.hist(y, bins=100, log=True)
				plt.title(f'{station} and {col}')
				plt.show()
			i+=1

		gc.collect()

# smoothed_median = interp1d(x, y_median, kind='cubic')
# smoothed_mean = interp1d(x, y_mean, kind='cubic')
# smoothed_topq = interp1d(x, y_topq, kind='cubic')
# smoothed_bottomq = interp1d(x, y_bottomq, kind='cubic')

# x_interp = np.linspace(min(x), max(x), num=100)
# y_interp_median = smoothed_median(x_interp)
# y_interp_mean = smoothed_mean(x_interp)
# y_interp_topq = smoothed_topq(x_interp)
# y_interp_bottomq = smoothed_bottomq(x_interp)

# ax1.plot(x_interp, y_interp_median, '-', label='median')
# ax1.plot(x_interp, y_interp_mean, '-', label='mean')
# ax1.plot(x_interp, y_interp_topq, '-', label='75th perc.')
# ax1.plot(x_interp, y_interp_bottomq, '-', label='25th perc.')
fig = plt.figure(figsize=(25,18))

ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)

ax1.scatter(x=x, y=y_median, label='median')
ax2.scatter(x=x, y=y_mean, label='mean', color='orange')
ax3.scatter(x=x, y=y_topq, label='75th perc.', color='green')
ax4.scatter(x=x, y=y_bottomq, label='25th perc.', color='red')
ax5.scatter(x=x, y=y_std, label='standard dev', color='purple')


# for i in range(len(x)):
# 	x_temp = [x[i]]*len(extreme_values[i])
# 	ax2.scatter(x=x_temp, y=extreme_values[i], color='red', label='99.9th perc values')

# Set y-axis label for significant points
ax1.set_xlabel('Distance (km)')
ax2.set_xlabel('Distance (km)')
ax3.set_xlabel('Distance (km)')
ax4.set_xlabel('Distance (km)')
ax5.set_xlabel('Distance (km)')

ax1.set_ylabel('Difference (dB/dt)')
ax2.set_ylabel('Difference (dB/dt)')
ax3.set_ylabel('Difference (dB/dt)')
ax4.set_ylabel('Difference (dB/dt)')
ax5.set_ylabel('Difference (dB/dt)')


ax1.set_title('Median')
ax2.set_title('Mean')
ax3.set_title('75th Percentile')
ax4.set_title('25th Percentile')
ax5.set_title('Standard Deviation')

# # Set y-axis label for extreme values
# ax2.set_ylabel('Difference (dB/dt)', color='r')

# Set the legend
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
plt.savefig('difference_and_distance_boxplots.png')

fig = plt.figure(figsize=(25,18))
ax1 = plt.subplot(111)
for i in range(len(x)):
	x_temp = [x[i]]*len(extreme_values[i])
	ax1.scatter(x=x_temp, y=extreme_values[i], color='purple')
ax1.set_xlabel('Distance (km)')
ax1.set_ylabel('Difference (dB/dt)')
ax1.set_title('Extreme Values (>99.9th percentile)')
plt.savefig('difference_and_distance_extreme_values.png')

print('Starting DBSCAN median....')
db_median_model = GaussianMixture(n_components=2, covariance_type='full')
data_median = pd.DataFrame({'x':x,'y':y_median,'station_0':station_0,'station_1':station_1}).dropna()
db_median_input = data_median[['x','y']].values
db_median_model.fit(db_median_input)
median_labels = db_median_model.predict(db_median_input)
data_median['cluster'] = median_labels

data_mean = pd.DataFrame({'x':x,'y':y_mean,'station_0':station_0,'station_1':station_1}).dropna()
db_mean_input = data_mean[['x','y']].values
print('Starting DBSCAN mean....')
db_mean_model = GaussianMixture(n_components=2, covariance_type='full')
db_mean_model.fit(db_mean_input)
mean_labels = db_mean_model.predict(db_mean_input)
data_mean['cluster'] = mean_labels

data_topq = pd.DataFrame({'x':x,'y':y_topq,'station_0':station_0,'station_1':station_1}).dropna()
db_topq_input = data_topq[['x','y']].values
print('Starting DBSCAN topq....')
db_topq_model = GaussianMixture(n_components=2, covariance_type='full')
db_topq_model.fit(db_mean_input)
topq_labels = db_topq_model.predict(db_topq_input)
data_topq['cluster'] = topq_labels

data_bottomq = pd.DataFrame({'x':x,'y':y_bottomq,'station_0':station_0,'station_1':station_1}).dropna()
db_bottomq_input = data_bottomq[['x','y']].values
print('Starting DBSCAN bottomq....')
db_bottomq_model = GaussianMixture(n_components=2, covariance_type='full')
db_bottomq_model.fit(db_bottomq_input)
bottomq_labels = db_bottomq_model.predict(db_bottomq_input)
data_bottomq['cluster'] = bottomq_labels

print('Starting DBSCAN std....')
db_std_model = GaussianMixture(n_components=2, covariance_type='full')
data_std = pd.DataFrame({'x':x,'y':y_std,'station_0':station_0,'station_1':station_1}).dropna()
db_std_input = data_std[['x','y']].values
db_std_model.fit(db_std_input)
std_labels = db_std_model.predict(db_std_input)
data_std['cluster'] = std_labels


fig = plt.figure(figsize=(25,18))
ax1 = plt.subplot(321)
ax2 = plt.subplot(322)
ax3 = plt.subplot(323)
ax4 = plt.subplot(324)
ax5 = plt.subplot(325)

ax1.scatter(x=data_median['x'], y=data_median['y'], label='median', c=median_labels)
ax2.scatter(x=data_mean['x'], y=data_mean['y'], label='mean', c=mean_labels)
ax3.scatter(x=data_topq['x'], y=data_topq['y'], label='75th perc.', c=topq_labels)
ax4.scatter(x=data_bottomq['x'], y=data_bottomq['y'], label='25th perc.', c=bottomq_labels)
ax5.scatter(x=data_std['x'], y=data_std['y'], label='STD', c=std_labels)

# Set y-axis label for significant points
ax1.set_xlabel('Distance (km)')
ax2.set_xlabel('Distance (km)')
ax3.set_xlabel('Distance (km)')
ax4.set_xlabel('Distance (km)')
ax5.set_xlabel('Distance (km)')

ax1.set_ylabel('Difference (nT/min)')
ax2.set_ylabel('Difference (nt/min)')
ax3.set_ylabel('Difference (nT/min)')
ax4.set_ylabel('Difference (nT/min)')
ax5.set_ylabel('Difference (nT/min)')

ax1.set_title('Median')
ax2.set_title('Mean')
ax3.set_title('75th Percentile')
ax4.set_title('25th Percentile')
ax5.set_title('STD')

# Set the legend
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
plt.savefig('difference_and_distance_clusters.png')

data_median.reset_index(inplace=True, drop=True)
data_mean.reset_index(inplace=True, drop=True)
data_topq.reset_index(inplace=True, drop=True)
data_bottomq.reset_index(inplace=True, drop=True)
data_std.reset_index(inplace=True, drop=True)

data_median.to_feather('outputs/data_median.feather')
data_mean.to_feather('outputs/data_mean.feather')
data_topq.to_feather('outputs/data_topq.feather')
data_bottomq.to_feather('outputs/data_bottomq.feather')
data_std.to_feather('outputs/data_std.feather')





