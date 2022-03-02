import pandas as pd
import math
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


col_list = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']
data = pd.read_csv('flights.csv', usecols = col_list)
airports = pd.read_csv('airports.csv')
    
cols = ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
data['ORIGIN-DESTINATION'] =  data[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

delay = {}
canceled = {}

#separately counting the number of delays, number of 30+ minutes delay and the cancelation of flights in regard to origin-destination pairs

for i in range(len(data['DEPARTURE_DELAY'])):
    if data['DEPARTURE_DELAY'][i] > 30:
        if data['ORIGIN-DESTINATION'][i] in delay.keys():
            delay[data['ORIGIN-DESTINATION'][i]] += 1
        else:
            delay[data['ORIGIN-DESTINATION'][i]] = 1
    
    if math.isnan(data['DEPARTURE_DELAY'][i]) == True:
        if data['ORIGIN-DESTINATION'][i] in canceled.keys():
            canceled[data['ORIGIN-DESTINATION'][i]] += 1
        else:
            canceled[data['ORIGIN-DESTINATION'][i]] = 1

weights_canceled = [x for x in canceled.values()]
origin = []
destination = []
for key in canceled.keys():
    splited = key.split('_')
    origin.append(splited[0])
    destination.append(splited[1])

for i in range(len(destination)):
    for j in range(len(airports['LATITUDE'])):
        if destination[i] == airports['IATA_CODE'][j]:
            destination[i] = (airports['LATITUDE'][j], airports['LONGITUDE'][j])
        if origin[i] == airports['IATA_CODE'][j]:
            origin[i] = (airports['LATITUDE'][j], airports['LONGITUDE'][j])

weights_canceled_for_sorting = [[] for i in range(len(weights_canceled))]
for i in range(len(weights_canceled)):
    weights_canceled_for_sorting[i].append(weights_canceled[i])
    weights_canceled_for_sorting[i].append(i)

weights_canceled_for_sorting = sorted(weights_canceled_for_sorting, key = lambda x : x[0], reverse = True)
w_c_norm = [float(i)/max(weights_canceled) for i in weights_canceled]

#ploting map of USA
#setting background with details

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.stock_img()
ax.set_extent([-129.768448, -67.366108,49.868420, 24.249941], ccrs.PlateCarree())

for i in range(len(weights_canceled)):
    plt.plot([origin[i][1], destination[i][1]], [origin[i][0], destination[i][0]], color='blue', linewidth=w_c_norm[i], marker='o',transform=ccrs.PlateCarree())
plt.close()

plt.figure(figsize=(17,17))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.stock_img()
ax.set_extent([-129.768448, -67.366108,49.868420, 24.249941], ccrs.PlateCarree())

for i in range(20):
    index = weights_canceled_for_sorting[i][1]
    for j in range(len(airports['LATITUDE'])):
        if origin[index][1] == airports['LONGITUDE'][j] and origin[index][0] == airports['LATITUDE'][j] and airports['CITY'][j] != 'Oakland':
            plt.text(origin[index][1]-1, origin[index][0]-1, airports['CITY'][j],
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
        if destination[index][1] == airports['LONGITUDE'][j] and destination[index][0] == airports['LATITUDE'][j] and airports['CITY'][j] != 'Oakland':
            plt.text(destination[index][1]-1, destination[index][0]-1, airports['CITY'][j],
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
        if airports['CITY'][j] == 'Oakland':
            plt.text(airports['LONGITUDE'][j]+1, airports['LATITUDE'][j]+1, airports['CITY'][j],
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
    plt.plot([origin[index][1], destination[index][1]], [origin[index][0], destination[index][0]], color='blue', linewidth=w_c_norm[index], marker='o',transform=ccrs.PlateCarree())
plt.savefig('First 20 cancelations')
plt.close()

o_d_most_cancelation = []
o_d_least_cancelation = []
index_most = weights_canceled_for_sorting[0][1]
index_least = weights_canceled_for_sorting[-1][1]
for j in range(len(airports['LATITUDE'])):
    if origin[index_most][1] == airports['LONGITUDE'][j] and origin[index_most][0] == airports['LATITUDE'][j]:
        o_d_most_cancelation.append(airports['AIRPORT'][j])
        o_d_most_cancelation.append(airports['CITY'][j])
    if destination[index_most][1] == airports['LONGITUDE'][j] and destination[index_most][0] == airports['LATITUDE'][j]:
        o_d_most_cancelation.append(airports['AIRPORT'][j])
        o_d_most_cancelation.append(airports['CITY'][j])
    if origin[index_least][1] == airports['LONGITUDE'][j] and origin[index_least][0] == airports['LATITUDE'][j]:
        o_d_least_cancelation.append(airports['AIRPORT'][j])
        o_d_least_cancelation.append(airports['CITY'][j])
    if destination[index_least][1] == airports['LONGITUDE'][j] and destination[index_least][0] == airports['LATITUDE'][j]:
        o_d_least_cancelation.append(airports['AIRPORT'][j])
        o_d_least_cancelation.append(airports['CITY'][j])
o_d_most_cancelation.append(weights_canceled_for_sorting[0][0])
o_d_least_cancelation.append(weights_canceled_for_sorting[-1][0])
print(o_d_most_cancelation)
print(o_d_least_cancelation)

counter = 0
for i in weights_canceled:
    if i == 1:
        counter += 1
print(counter)

weights_delay = [x for x in delay.values()]
origin_delay = []
destination_delay = []
for key in delay.keys():
    splited = key.split('_')
    origin_delay.append(splited[0])
    destination_delay.append(splited[1])

for i in range(len(destination_delay)):
    for j in range(len(airports['LATITUDE'])):
        if destination_delay[i] == airports['IATA_CODE'][j]:
            destination_delay[i] = (airports['LATITUDE'][j], airports['LONGITUDE'][j])
        if origin_delay[i] == airports['IATA_CODE'][j]:
            origin_delay[i] = (airports['LATITUDE'][j], airports['LONGITUDE'][j])

w_d_norm = [float(i)/max(weights_delay) for i in weights_delay]

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.stock_img()
ax.set_extent([-129.768448, -67.366108,49.868420, 24.249941], ccrs.PlateCarree())

for i in range(len(weights_delay)):
    plt.plot([origin_delay[i][1], destination_delay[i][1]], [origin_delay[i][0], destination_delay[i][0]], color='blue', linewidth=w_d_norm[i], marker='o',transform=ccrs.PlateCarree())
plt.close()

weights_delayed_for_sorting = [[] for i in range(len(weights_delay))]
for i in range(len(weights_delay)):
    weights_delayed_for_sorting[i].append(weights_delay[i])
    weights_delayed_for_sorting[i].append(i)

weights_delayed_for_sorting = sorted(weights_delayed_for_sorting, key = lambda x : x[0], reverse = True)

plt.figure(figsize=(17,17))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.stock_img()
ax.set_extent([-129.768448, -67.366108,49.868420, 24.249941], ccrs.PlateCarree())

airport_names = []
for i in range(10):
    index = weights_delayed_for_sorting[i][1]
    for j in range(len(airports['LATITUDE'])):
        if origin_delay[index][1] == airports['LONGITUDE'][j] and origin_delay[index][0] == airports['LATITUDE'][j] and airports['CITY'][j] not in airport_names:
            plt.text(origin_delay[index][1]-1, origin_delay[index][0]-1, airports['CITY'][j],
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            airport_names.append(airports['CITY'][j])
        if destination_delay[index][1] == airports['LONGITUDE'][j] and destination_delay[index][0] == airports['LATITUDE'][j] and airports['CITY'][j] not in airport_names:
            plt.text(destination_delay[index][1]-1, destination[index][0]-1, airports['CITY'][j],
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            airport_names.append(airports['CITY'][j])
    plt.plot([origin_delay[index][1], destination_delay[index][1]], [origin_delay[index][0], destination_delay[index][0]], color='blue', linewidth=w_d_norm[index], marker='o',transform=ccrs.PlateCarree())
plt.savefig('First 10 delays')
plt.close()

o_d_most_delay = []
o_d_least_delay = []
index_most = weights_delayed_for_sorting[0][1]
index_least = weights_delayed_for_sorting[-1][1]
for j in range(len(airports['LATITUDE'])):
    if origin_delay[index_most][1] == airports['LONGITUDE'][j] and origin_delay[index_most][0] == airports['LATITUDE'][j]:
        o_d_most_delay.append(airports['AIRPORT'][j])
        o_d_most_delay.append(airports['CITY'][j])
    if destination_delay[index_most][1] == airports['LONGITUDE'][j] and destination_delay[index_most][0] == airports['LATITUDE'][j]:
        o_d_most_delay.append(airports['AIRPORT'][j])
        o_d_most_delay.append(airports['CITY'][j])
    if origin_delay[index_least][1] == airports['LONGITUDE'][j] and origin_delay[index_least][0] == airports['LATITUDE'][j]:
        o_d_least_delay.append(airports['AIRPORT'][j])
        o_d_least_delay.append(airports['CITY'][j])
    if destination_delay[index_least][1] == airports['LONGITUDE'][j] and destination_delay[index_least][0] == airports['LATITUDE'][j]:
        o_d_least_delay.append(airports['AIRPORT'][j])
        o_d_least_delay.append(airports['CITY'][j])
o_d_most_delay.append(weights_delayed_for_sorting[0][0])
o_d_least_delay.append(weights_delayed_for_sorting[-1][0])
print(o_d_most_delay)
print(o_d_least_delay)

counter = 0
for i in weights_delay:
    if i == 1:
        counter += 1
print(counter)