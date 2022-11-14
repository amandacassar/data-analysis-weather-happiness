#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import os, shutil
import requests, re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statistics import variance


# In[2]:



##########################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   PART 1   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################################################


# In[3]:


######################################################################################
##################    Variables that can be altered by the user:    ##################
##################     'selected_features' and 'num_clusters'       ##################
######################################################################################

# Select the features to be used for clustering the data - choose from 'TMAX', 'TMIN', 'AF', 'RAIN', 'SUN'
# Note - the scatterplots (later) suggest that TMIN and AF have an almost linear relationship with TMAX,
# therefore, 'TMAX', 'RAIN' and 'SUN' are the default selection
selected_features = ['TMAX','RAIN','SUN']

# Determine the number of clusters - must be an integer
# Note - the dendrogram (later) suggests that optimal number of clusters is 3
num_clusters = 3


# In[4]:


# Read the 'stations' text file which contains the list of stations
stations_file = open('Data/stations.txt', 'r')
stations = stations_file.readlines()

# Store the stations' names in a list
stations_list = []
for station in stations:
    # Strip function removes any white spaces
    stations_list.append(station.strip())

stations_list


# In[5]:


# Check if the 'Weather' folder exists and if so delete it
if os.path.exists('Data/Weather'):
    shutil.rmtree('Data/Weather')
    
# Then create a new 'Weather' folder to contain the new data
os.makedirs('Data/Weather')


# In[6]:


# Obtain the weather data for every station - download data from each station's url
# Save the data in the 'Data' folder
for station in stations_list:
    url = 'http://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/'+station+'data.txt'
    r = requests.get(url, allow_redirects=True)
    open('Data/Weather/'+station+'.txt', 'wb').write(r.content)


# In[7]:


# Store the data of all the stations in a dictionary
all_data_dict = {}

# Read the data from all the textfiles
for station in stations_list:
    file = open('Data/Weather/'+station+'.txt', 'r')
    
    # Store each line (record) in a list - i.e., each list will consist of 7 values (year, month, tmax, tmin, af, rain, sun)
    features_list = []
    for line in file:
        strip_line = line.strip()
        features_line = strip_line.split()
        # Do not include the initial data and headers
        try:
            # Check whether the first string can be converted to a number - every weather data line starts with a number (year)
            int(features_line[0])
            # Also, remove any irrelevant text next to the features (7 features => 7 numbers)
            if (len(features_line) > 7):
                delete_data = len(features_line) - 7
                del features_line[-delete_data:]                
            features_list.append(features_line)
        except:
            continue
    
    # Close the file
    file.close()

    # Store the list of all the features for the current station in the dictionary, with the station name as key
    all_data_dict[station] = features_list

all_data_dict


# In[8]:


# Determine the first year in which all stations had records taken in January
# so that every station has the same number of instances
years_list = []
for station in stations_list:
    # Obtain the first instance when month is January, for the current station
    month = 0
    index = 0
    while month != 1:
        month = int(all_data_dict[station][index][1])
        if (month == 1):
            # When the month is 1 (i.e. January), obtain the year corresponding to the current month, for the current station
            # Convert year value to integers to enable comparison
            years_list.append(int(all_data_dict[station][index][0]))
        # Increment the index variable by 1 with every while loop
        index += 1   

# Select the biggest value from the list of years - being the first year when all stations had records starting from January
first_year = max(years_list)

print(years_list)
first_year


# In[9]:


# Create a list which excludes the data that is no longer needed - i.e., exclude records of years before the 'first_year'
reduced_list = []
for station in stations_list:    
    # Loop through the length of the current dictionary values (lists)
    for n in range(len(all_data_dict[station])):
        curr_record = []
        # If the year for the current element in the list is >= first_year, then include this list element in the final
        # features list of the current station
        if int(all_data_dict[station][n][0]) >= first_year:
            # Include the station name with each record, to later convert the list into a dataframe
            curr_record.append(station)
            for i in range(len(all_data_dict[station][n])):
                # First check if data contains any special characters or text
                # Remove all characters that are not numeric, or decimal point, or '-'
                curr_feature = re.sub('[^0-9 . -]+', '', all_data_dict[station][n][i]) 
                curr_record.append(curr_feature)
            # Store the features values (lists) in the new dictionary
            reduced_list.append(curr_record)

reduced_list


# In[10]:


# Create a pandas dataframe from the data in processed_dict, to facilitate pre-processing tasks
df = pd.DataFrame(reduced_list, columns = ['STATION','YEAR','MONTH','TMAX','TMIN','AF','RAIN','SUN'])
df


# In[11]:


# Replace '---' values with NaN
df.replace('---', np.nan, inplace=True)


# In[12]:


# Check the percentage of missing values for each feature in the dataframe
print(df.isnull().sum()/len(df))


# In[13]:


# Convert strings to float
df['TMAX'] = df['TMAX'].astype(float)
df['TMIN'] = df['TMIN'].astype(float)
df['AF'] = df['AF'].astype(float)
df['RAIN'] = df['RAIN'].astype(float)
df['SUN'] = df['SUN'].astype(float)


# In[14]:


# To impute missing values, find the average value for every feature, for every month, for each station
# Store the averages in an a list of dictionaries - each feature includes a list of 12 mean values for a given station
means_list = []
for station in stations_list:
    means_dict = {}
    means_dict["station"] = station
    tmax_mean, tmin_mean, af_mean, rain_mean, sun_mean = ([] for i in range(5))
    for m in range(12):
        tmax = df['TMAX'].loc[df['MONTH']==str(m+1)].loc[df['STATION']==station]
        tmin = df['TMIN'].loc[df['MONTH']==str(m+1)].loc[df['STATION']==station]
        af = df['AF'].loc[df['MONTH']==str(m+1)].loc[df['STATION']==station]
        rain = df['RAIN'].loc[df['MONTH']==str(m+1)].loc[df['STATION']==station]
        sun = df['SUN'].loc[df['MONTH']==str(m+1)].loc[df['STATION']==station]
        # Obtain the mean value for every feature, for every month; ignore the NAN values
        tmax_mean.append(tmax.mean(skipna=True))
        tmin_mean.append(tmin.mean(skipna=True))
        af_mean.append(af.mean(skipna=True))
        rain_mean.append(rain.mean(skipna=True))
        sun_mean.append(sun.mean(skipna=True))
    # Store in a list the 12 mean values (one for each month) for the current station
    means_dict['tmax_mean'] = tmax_mean
    means_dict['tmin_mean'] = tmin_mean
    means_dict['af_mean'] = af_mean
    means_dict['rain_mean'] = rain_mean
    means_dict['sun_mean'] = sun_mean
    # Store the data (as a dictionary) about the current station in the list
    means_list.append(means_dict)


# In[15]:


# Impute missing values with the average value of the respective feature, for that particular month
s_index = 0
for station in stations_list:
    for m in range(12):
        # Obtain the mean for all the features of the current month for the current station
        tmax_impute = means_list[s_index]['tmax_mean'][m]
        tmin_impute = means_list[s_index]['tmin_mean'][m]
        af_impute = means_list[s_index]['af_mean'][m]
        rain_impute = means_list[s_index]['rain_mean'][m]
        sun_impute = means_list[s_index]['sun_mean'][m]
        # Replace NaN with mean values, and store new list values in a temporary variable
        df_temp = df.loc[df['MONTH']==str(m+1)].loc[df['STATION']==station].replace({'TMAX': np.nan}, tmax_impute)
        # Store the new data in the dataframe
        df.update(df_temp)
        # Repeat the last two processes for every feature
        df_temp = df.loc[df['MONTH']==str(m+1)].loc[df['STATION']==station].replace({'TMIN': np.nan}, tmin_impute)
        df.update(df_temp)
        df_temp = df.loc[df['MONTH']==str(m+1)].loc[df['STATION']==station].replace({'AF': np.nan}, af_impute)
        df.update(df_temp)
        df_temp = df.loc[df['MONTH']==str(m+1)].loc[df['STATION']==station].replace({'RAIN': np.nan}, rain_impute)
        df.update(df_temp)
        df_temp = df.loc[df['MONTH']==str(m+1)].loc[df['STATION']==station].replace({'SUN': np.nan}, sun_impute)
        df.update(df_temp)
    s_index += 1


# In[16]:


# Check the percentage of missing values for each feature in the dataframe
print(df.isnull().sum()/len(df))


# In[17]:


# Some station(s) did not have any records for Sun hours
# Since we cannot perform clustering with NaN values, the missing Sun hours will be imputed with the overall average Sun hours
# Calculate the Sun overall mean
sun_overall_mean = df['SUN'].mean(skipna=True)
# Obtain the indices of the rows with no Sun values, and drop (remove) these rows from the dataframe
null_list = df[df['SUN'].isnull()].index.tolist()

# Impute the missing Sun hours values with the overall mean
df.loc[null_list[0]:null_list[-1],'SUN'] = sun_overall_mean


# In[18]:


# Check the percentage of missing values for each feature in the dataframe
print(df.isnull().sum()/len(df))


# In[19]:


# Observe how the data is scattered on a scatter plot
# Plot scatter plots between each two features
# If any two features show a linear relationship, select only one of those two features
plt.scatter(df['TMAX'], df['TMIN'], s=100)
plt.title('TMAX vs TMIN')
plt.xlabel('TMAX')
plt.ylabel('TMIN')


# In[20]:


plt.scatter(df['TMAX'], df['RAIN'], s=100)
plt.title('TMAX vs RAIN')
plt.xlabel('TMAX')
plt.ylabel('RAIN')


# In[21]:


plt.scatter(df['TMAX'], df['AF'], s=100)
plt.title('TMAX vs AF')
plt.xlabel('TMAX')
plt.ylabel('AF')


# In[22]:


plt.scatter(df['TMAX'], df['SUN'], s=100)
plt.title('TMAX vs SUN')
plt.xlabel('TMAX')
plt.ylabel('SUN')


# In[23]:


plt.scatter(df['RAIN'], df['SUN'], s=100)
plt.title('RAIN vs SUN')
plt.xlabel('RAIN')
plt.ylabel('SUN')


# In[24]:


# TMIN and AF showed an almost linear relationship with TMAX, thus TMIN and AF will not be used for clustering the data
# Unless altered by the user, TMIN and AF are excluded from the 'selected_features' array


# In[25]:


# Create the array that will contain the data - include only the selected features
df_features = df[selected_features]

# Convert dataframe to array
weather_data_arr = df_features.to_numpy()


# In[26]:


# Plot the dendogram to visualise the data clustering and determine the number of clusters
plt.figure(figsize=(10,7))
dendro = dendrogram(linkage(weather_data_arr,method='ward'))
plt.show


# In[27]:


# Perform Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')

# Fit the model on the data and predict the clusters
fit = cluster.fit_predict(weather_data_arr)


# In[28]:


# Check the performance of Agglomerative clustering
silhouette_score(weather_data_arr,fit)


# In[29]:


# Plot the clusters
plt.scatter(weather_data_arr[fit==0,0], weather_data_arr[fit==0,1], s=100, c='yellow')
plt.scatter(weather_data_arr[fit==1,0], weather_data_arr[fit==1,1], s=100, c='red')
plt.scatter(weather_data_arr[fit==2,0], weather_data_arr[fit==2,1], s=100, c='blue')
plt.title('Agglomerative Clustering Before Scaling')


# In[30]:


# Perform K-means Clustering
kmeans = KMeans(n_clusters=num_clusters, init='random')

# Fit the data
kmeans.fit(weather_data_arr)
pred = kmeans.predict(weather_data_arr)


# In[31]:


# Check the performance of K-means clustering
silhouette_score(weather_data_arr, pred)


# In[32]:


# Normalise the data - also referred to as 'Scaling'
# The data is normalised so that all the data fits within a scale, and hence obtain better results
weather_scaled_arr = normalize(weather_data_arr)
weather_scaled_arr


# In[33]:


# Perform Agglomerative Clustering on the scaled data
cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
fit = cluster.fit_predict(weather_scaled_arr)


# In[34]:


# Check the performance of Agglomerative clustering on the scaled data
silhouette_score(weather_scaled_arr,fit)


# In[35]:


# Perform K-means clustering on the scaled data
kmeans = KMeans(n_clusters=num_clusters, init='random')
kmeans.fit(weather_scaled_arr)
pred = kmeans.predict(weather_scaled_arr)


# In[36]:


# Check the performance of K-means clustering on the scaled data
silhouette_score(weather_scaled_arr, pred)


# In[37]:


# Plot the clusters for the Agglomeratvie clustering algorithm
plt.scatter(weather_scaled_arr[fit==0,0], weather_scaled_arr[fit==0,1], s=100, c='yellow')
plt.scatter(weather_scaled_arr[fit==1,0], weather_scaled_arr[fit==1,1], s=100, c='red')
plt.scatter(weather_scaled_arr[fit==2,0], weather_scaled_arr[fit==2,1], s=100, c='blue')
plt.title('Agglomerative Clustering After Scaling')


# In[38]:


# Plot the clusters for the K-means clustering algorithm
plt.scatter(weather_scaled_arr[pred==0,0], weather_scaled_arr[pred==0,1], s=100, c='yellow')
plt.scatter(weather_scaled_arr[pred==1,0], weather_scaled_arr[pred==1,1], s=100, c='red')
plt.scatter(weather_scaled_arr[pred==2,0], weather_scaled_arr[pred==2,1], s=100, c='blue')
plt.title('K-means Clustering After Scaling')


# In[39]:


# Obtain the most frequent label for each station, to determine the cluster of each station
# When there is a change of station in the dataframe, calculate the most frequent cluster for that station
curr_station = df['STATION'][0] 
curr_clusters = []
clusters_dict = {}
counter = 0
for i in range(len(fit)):
    # If this is not the current element, compare with previous station name and obtain the cluster value
    if i < len(fit)-1:
        if curr_station == df['STATION'][i]:
            curr_clusters.append(fit[i])
        else:
            # If station changed:
            # 1. obtain the most frequent cluster value for the current station,
            # 2. update temporary variables,       
            clusters_dict[curr_station] = max(set(curr_clusters), key = curr_clusters.count)
            curr_clusters = []
            curr_station = df['STATION'][i]
    else:
        # If it is the last element, obtain the most frequent cluster value for the current station
        curr_clusters.append(fit[i])
        clusters_dict[curr_station] = max(set(curr_clusters), key = curr_clusters.count)

print(clusters_dict)


# In[40]:


# Group the stations of each cluster and print results
# Sort the dictionary by value (i.e. by station's cluster)
sorted_dict = dict(sorted(clusters_dict.items(), key=lambda item: item[1]))
curr_value = ''
for key in sorted_dict.keys():
    if curr_value != sorted_dict[key]:
        print('----------------------------')
        curr_value = sorted_dict[key]
    print(sorted_dict[key], ': ', key)


# In[41]:



##########################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   PART 2   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################################################


# In[42]:


##################################################################################################
##################          Variables that can be altered by the user:          ##################
##################  'exclude_stations', 'training_percentage' and 'classifier'  ##################
##################################################################################################

# Number of stations to exclude from the full stations list - must be an integer
exclude_stations = 5

# Two-fold validation - enter the size (%) of the training dataset (size of testing dataset = 100 - size of training dataset)
training_percentage = 80

# Classification algorithm to be used - choose from:
# 'KNN' - K-Nearest-Neighbor
# 'SVM' - Support Vector Machine
# 'NB' - Gaussian Naive Bayes
# 'DT' - Decision Tree
# 'RF' - Random Forest
# 'NN' - Neural Network (Multi-Layer Perceptron)
classifier = 'RF'


# In[43]:


# Obtain the latitude data from the textfiles
stations_latitude_dict = {}
for station in stations_list:
    file = open('Data/Weather/'+station+'.txt', 'r')
    
    # Store the station and respective latitude in a dictionary
    for line in file:
        # Extract the latitude data 
        if 'Lat' in line:
            strip_line = line.strip()
            split_line = strip_line.split()
            # Extract the text following the word 'Lat', split the extracted text, and get the first string
            latitude = strip_line.split('Lat',1)[1].split()[0]
            # Convert to integer
            latitude = float(latitude)
            stations_latitude_dict[station] = latitude

print(stations_latitude_dict)


# In[44]:


# Determine the north, centre and south latitudes
# Most northern latitude is 60.9, and most southern latitude is 49.9
max_lat = 60.9
min_lat = 49.9
# Calculate the lower boundary of the middle area (i.e., the boundary between centre and south)
mid_lower_lat = ((max_lat - min_lat) / 3) + min_lat
# Calculate the upper boundary of the middle area (i.e., the boundary between centre and north)
mid_upper_lat = max_lat - ((max_lat - min_lat) / 3)

print(mid_lower_lat)
print(mid_upper_lat)


# In[45]:


# Create a dictionary that holds the class of each station - whether it is in the north, centre, or south of UK
# 0 = south
# 1 = centre
# 2 = north
latitude_labels_dict = {}
for key in stations_latitude_dict:
    curr_lat = stations_latitude_dict[key]
    if (curr_lat < mid_lower_lat):
        latitude_labels_dict[key] = 0
    elif (curr_lat > mid_upper_lat):
        latitude_labels_dict[key] = 2
    else:
        latitude_labels_dict[key] = 1
    # Print current result - for checking purposes
    #print(str(latitude_values[key]) + ' = ' + str(latitude_labels[key]))

print(latitude_labels_dict)


# In[46]:


# Include the latitude data in the weather dataframe
all_latitude_labels = []
for i in range(len(df['STATION'])):
    curr_lat_label = latitude_labels_dict[df['STATION'][i]]
    all_latitude_labels.append(curr_lat_label)
df['LAT LABEL'] = all_latitude_labels


# In[47]:


# Exclude the last 5 stations, as required for this project (default = 5 stations, but user can change this number)
reduced_stations_list = stations_list[:-exclude_stations]
# Get the name of the last selected station
last_station = list(reduced_stations_list)[-1]
# Get the index where the last instance of the 'last_station' resides in the dataframe
# First store the dataframe's station data in a list
# Then determine where in that list resides the last instance of the 'last_station'
all_stations_in_df = df['STATION']
last_index = np.where(all_stations_in_df == last_station)[0][-1]

# Create a new dataframe to contain the reduced data (i.e., not including the excluded stations)
reduced_df = df[:last_index]
#reduced_df

# Create two arrays - one will hold the weather data, and one will hold the latitude labels
data_arr = reduced_df[selected_features]
labels_arr = reduced_df['LAT LABEL']

print(data_arr)
print(labels_arr)


# In[48]:


# Split the data into training and testing datasets
# Split in the ratio selected by the user earlier
t_size = (100 - training_percentage) / 100
train_data, test_data, train_labels, test_labels = train_test_split(data_arr, labels_arr, test_size=t_size)


# In[49]:


if classifier == 'KNN':
    # Number of nearest neighbours set to 5 (default value); the default values for weights and other parameters are used
    clfr = KNeighborsClassifier(5)
elif classifier == 'SVM':
    # Linear SVM with regularisation parameter set as 0.05
    clfr = SVC(kernel='linear', C=0.05)
elif classifier == 'NB':
    # Naive Bayes - parameteres set as default - no adjusted probabilites and no variance stability
    clfr = GaussianNB()
elif classifier == 'DT':
    # Decision tree - maximum depth of tree set to 5
    clfr = DecisionTreeClassifier(max_depth=5)
elif classifier == 'RF':
    # Random forest - maximum depth of tree set to 5 and number of trees in the forest set to 150
    clfr = RandomForestClassifier(max_depth=5, n_estimators=150)
elif classifier == 'NN':
    # Neural network (multi-layer perceptron) - logistic (sigmoid) function and maximum iterations set as 300
    clfr = MLPClassifier(activation='logistic', max_iter=300)

clfr.fit(train_data, train_labels)
accuracy = clfr.score(test_data, test_labels)

accuracy


# In[50]:



##########################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   PART 3   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
##########################################################################


# In[51]:


# Download the 'personal well-being' datasets for 2011-15, and store them in the 'Happiness' folder
urls = []
urls.append('https://www.ons.gov.uk/file?uri=%2fpeoplepopulationandcommunity%2fwellbeing%2fdatasets%2fpersonalwellbeingestimatesgeographicalbreakdown%2f201112/20112012referencetabletcm77332122.xls')
urls.append('https://www.ons.gov.uk/file?uri=%2fpeoplepopulationandcommunity%2fwellbeing%2fdatasets%2fpersonalwellbeingestimatesgeographicalbreakdown%2f201213/20122013referencetable_tcm77-332116.xls')
urls.append('https://www.ons.gov.uk/file?uri=%2fpeoplepopulationandcommunity%2fwellbeing%2fdatasets%2fpersonalwellbeingestimatesgeographicalbreakdown%2f201314/referencetable1geographicalbreakdown_tcm77-378058.xls')
urls.append('https://www.ons.gov.uk/file?uri=%2fpeoplepopulationandcommunity%2fwellbeing%2fdatasets%2fpersonalwellbeingestimatesgeographicalbreakdown%2f201415/geographicbreakdownreferencetable_tcm77-417203.xls')

if os.path.exists('Data/Happiness'):
    shutil.rmtree('Data/Happiness')
os.makedirs('Data/Happiness')

for i in range(len(urls)):
    r = requests.get(urls[i], allow_redirects=True)
    open('Data/Happiness/'+str(i)+'.xls', 'wb').write(r.content)


# In[52]:


# Store the data within the 'happiness' tab in a dataframe
happiness_df = pd.DataFrame()
periods_list = ['2011-12', '2012-13', '2013-14', '2014-15']
for i in range(len(urls)):
    file = pd.ExcelFile('Data/Happiness/'+str(i)+'.xls')
    curr_df = pd.read_excel(file, 'Happiness')
    # When reading the data, skip the first few rows which contain irrelevant information
    # Determine the row that contains the headers
    for row in range(curr_df.shape[0]):        
        # Convert current cell to string
        curr_cell = str(curr_df.iat[row,0])
        # iat[] method is used to return data in the dataframe at the current location
        if ('Codes' in curr_cell):
            row_start = row
            break
    # Exclude the rows before the headers
    curr_df = curr_df.loc[row_start+1:]
    # Rename the columns
    curr_df.rename(columns={'Unnamed: 1': 'Area Names', 'Unnamed: 4': 'Low', 'Unnamed: 5': 'Medium'}, inplace=True)
    curr_df.rename(columns={'Unnamed: 6': 'High', 'Unnamed: 7': 'Very High', 'Unnamed: 8': 'Average'}, inplace=True)
    # Keep only the necessary columns (B, E-I)
    curr_df = curr_df.iloc[:, [1, 4, 5, 6, 7, 8]]
    # Include the time period
    curr_df['Time Period'] = periods_list[i]
    # Add current dataframe to the previous dataframe, to have all the years' data in one dataframe
    happiness_df = pd.concat([happiness_df, curr_df])

happiness_df


# In[53]:


# Obtain the regions that are included in the 'regions' text file
regions_file = open('Data/regions.txt', 'r')
regions = regions_file.readlines()

# Store the regions' names in a dictionary
# We will use the latitude to combine happiness data with weather data
# Thus, store the region name and the latitude in the dictionary
regions_latitude_dict = {}
for region in regions:
    # Strip function removes any white spaces
    line = region.strip()
    split_line = line.split(',')     
    region = split_line[1]
    latitude = float(split_line[2])
    regions_latitude_dict[region] = latitude

# Order the dictionary by the key
regions_latitude_dict = OrderedDict(sorted(regions_latitude_dict.items()))
regions_latitude_dict


# In[54]:


# In the happiness dataframe, include only the selected regions
regions_list = []
for key in regions_latitude_dict:
    regions_list.append(key)

# Remove any white space or numeric values in the dataframe in the 'Area names' columns, to allow comparison with 'regions_list'
happiness_df['Area Names'] = happiness_df['Area Names'].str.strip()
happiness_df['Area Names'].replace('[0-9]+', '', regex=True, inplace=True)
# Update the dataframe to contain only the records of the selected regions
happiness_df = happiness_df.loc[happiness_df['Area Names'].isin(regions_list)]
# Sort the dataframe by the area names and time period
#happiness_df = happiness_df.sort_values('Area Names')
happiness_df = happiness_df.sort_values(by = ['Area Names', 'Time Period'], ascending = [True, True], na_position = 'first')
# Reset the indices so that we can concatenate this dataframe with the weather dataframe later
happiness_df.reset_index(drop=True, inplace=True)
happiness_df


# In[55]:


# Determine the closest station for every region, by using the the latitude values
region_station_list = []
for region in regions_latitude_dict:
    region_lat = regions_latitude_dict[region]
    # The station closest to the current region is the station with minimum difference in latitude
    closest_station = min(stations_latitude_dict.items(), key=lambda x: abs(region_lat - x[1]))
    curr_list = []
    curr_list.append(region)
    curr_list.append(closest_station[0])
    region_station_list.append(curr_list)
region_station_df = pd.DataFrame(region_station_list, columns=['REGION', 'STATION'])
region_station_df


# In[56]:


# Obtain the weather data for all the years within the happiness dataframe
# The happiness dataframe's years always start in April and end in March of the consecutive year
years = ['2011','2012','2013','2014','2015']
weather_df = pd.DataFrame()
for i in range(len(periods_list)):    
    m1 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='4']
    m2 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='5']
    m3 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='6']
    m4 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='7']
    m5 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='8']
    m6 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='9']
    m7 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='10']
    m8 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='11']
    m9 = df.loc[df['YEAR']==years[i]].loc[df['MONTH']=='12']
    m10 = df.loc[df['YEAR']==years[i+1]].loc[df['MONTH']=='1']
    m11 = df.loc[df['YEAR']==years[i+1]].loc[df['MONTH']=='2']
    m12 = df.loc[df['YEAR']==years[i+1]].loc[df['MONTH']=='3']    
    this_df = pd.concat([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12])
    
    period = []
    for j in range(len(this_df)):
        period.append(periods_list[i])
    this_df['PERIOD'] = period
    # Add current dataframe to the previous dataframe, to have all the years' data in one dataframe
    weather_df = pd.concat([weather_df, this_df])
weather_df


# In[57]:


# Since the happiness data consists of yearly average values, we obtain the yearly average values for the weather data too
# Obtain the average weather features values of the stations that correspond to the 12 regions 
weather_mean_list = []
for station in region_station_df['STATION']:
    for i in range(len(periods_list)):
        curr_tmax = weather_df['TMAX'].loc[weather_df['STATION']==station].loc[weather_df['PERIOD']==periods_list[i]]
        curr_rain = weather_df['RAIN'].loc[weather_df['STATION']==station].loc[weather_df['PERIOD']==periods_list[i]]
        curr_sun = weather_df['SUN'].loc[weather_df['STATION']==station].loc[weather_df['PERIOD']==periods_list[i]]
        curr_region = region_station_df['REGION'].loc[region_station_df['STATION']==station]
        curr_mean_list = []
        curr_mean_list.append(station)
        curr_mean_list.append(curr_tmax.mean(skipna=True))
        curr_mean_list.append(curr_rain.mean(skipna=True))
        curr_mean_list.append(curr_sun.mean(skipna=True))
        curr_mean_list.append(curr_region.values[0])
        curr_mean_list.append(periods_list[i])
        weather_mean_list.append(curr_mean_list)

weather_mean_df = pd.DataFrame(weather_mean_list, columns=['STATION', (*selected_features), 'REGION', 'PERIOD'])
weather_mean_df
# Note that 'cardiff' and 'bradford' stations are repeated - due to the rough latitude method used to estimate closest station
# 'cardiff' resulted closest to London and Wales (Wales is replaced with London below)
# 'bradford' resulted closest to North West and Yorkshire (Yorkshir is replaced with North West below)


# In[58]:


# Join the happiness data and the weather data
# The order of the rows in 'weather_mean_df' matches the order of the rows in 'happiness_df'
happiness_weather_df = pd.concat([happiness_df, weather_mean_df], axis=1)
happiness_weather_df


# In[59]:


# Convert objects to float
happiness_weather_num_df = pd.DataFrame()
happiness_weather_num_df['Low'] = happiness_weather_df['Low'].astype(float)
happiness_weather_num_df['Medium'] = happiness_weather_df['Medium'].astype(float)
happiness_weather_num_df['High'] = happiness_weather_df['High'].astype(float)
happiness_weather_num_df['Very High'] = happiness_weather_df['Very High'].astype(float)
happiness_weather_num_df['Average'] = happiness_weather_df['Average'].astype(float)
happiness_weather_num_df['TMAX'] = happiness_weather_df['TMAX']
happiness_weather_num_df['RAIN'] = happiness_weather_df['RAIN']
happiness_weather_num_df['SUN'] = happiness_weather_df['SUN']
happiness_weather_num_df.dtypes


# In[60]:


# Select the numerical variables of interest
happiness_weather_num_df = happiness_weather_num_df.select_dtypes(include=np.number)
happiness_weather_num_df


# In[61]:


# Apply the correlation function
corr = happiness_weather_num_df.corr()
corr


# In[62]:


# Repeat the analysis but using only one year's data - period 2014-15
h_w_one_year_df = happiness_weather_df.loc[happiness_weather_df['PERIOD']=='2014-15']
h_w_one_year_df


# In[63]:


# Convert objects to float
h_w_one_year_num_df = pd.DataFrame()
h_w_one_year_num_df['Low'] = h_w_one_year_df['Low'].astype(float)
h_w_one_year_num_df['Medium'] = h_w_one_year_df['Medium'].astype(float)
h_w_one_year_num_df['High'] = h_w_one_year_df['High'].astype(float)
h_w_one_year_num_df['Very High'] = h_w_one_year_df['Very High'].astype(float)
h_w_one_year_num_df['Average'] = h_w_one_year_df['Average'].astype(float)
h_w_one_year_num_df['TMAX'] = h_w_one_year_df['TMAX']
h_w_one_year_num_df['RAIN'] = h_w_one_year_df['RAIN']
h_w_one_year_num_df['SUN'] = h_w_one_year_df['SUN']

# Select the numerical variables of interest
h_w_one_year_num_df = h_w_one_year_num_df.select_dtypes(include=np.number)

# Apply the correlation function
corr = h_w_one_year_num_df.corr()
corr


# In[64]:


# Additional observations below


# In[65]:


# Observe the variance in the happiness data - high variance may cause overfitting
print(variance(happiness_df['Average']))
print(variance(happiness_df['Low']))
print(variance(happiness_df['Medium']))
print(variance(happiness_df['High']))
print(variance(happiness_df['Very High']))


# In[66]:


# Scatter plots between all stations features and region features, for all the years
# Change the default figure parameters to make bigger plots
plt.rcParams['figure.figsize'] = [20, 28]

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(5, 3)

# Determine the several x and y axis
x1 = happiness_weather_df['TMAX']
x2 = happiness_weather_df['RAIN']
x3 = happiness_weather_df['SUN']
y1 = happiness_weather_df['Average']
y2 = happiness_weather_df['Low']
y3 = happiness_weather_df['Medium']
y4 = happiness_weather_df['High']
y5 = happiness_weather_df['Very High']

axis[0, 0].scatter(x1,y1)
axis[0, 0].set_title('TMAX - Average Happiness')
axis[0, 1].scatter(x2,y1)
axis[0, 1].set_title('RAIN - Average Happiness')
axis[0, 2].scatter(x3,y1)
axis[0, 2].set_title('SUN - Average Happiness')
axis[1, 0].scatter(x1,y2)
axis[1, 0].set_title('TMAX - Low Happiness')
axis[1, 1].scatter(x2,y2)
axis[1, 1].set_title('RAIN - Low Happiness')
axis[1, 2].scatter(x3,y2)
axis[1, 2].set_title('SUN - Low Happiness')
axis[2, 0].scatter(x1,y3)
axis[2, 0].set_title('TMAX - Medium Happiness')
axis[2, 1].scatter(x2,y3)
axis[2, 1].set_title('RAIN - Medium Happiness')
axis[2, 2].scatter(x3,y3)
axis[2, 2].set_title('SUN - Medium Happiness')
axis[3, 0].scatter(x1,y4)
axis[3, 0].set_title('TMAX - High Happiness')
axis[3, 1].scatter(x2,y4)
axis[3, 1].set_title('RAIN - High Happiness')
axis[3, 2].scatter(x3,y4)
axis[3, 2].set_title('SUN - High Happiness')
axis[4, 0].scatter(x1,y5)
axis[4, 0].set_title('TMAX - Very High Happiness')
axis[4, 1].scatter(x2,y5)
axis[4, 1].set_title('RAIN - Very High Happiness')
axis[4, 2].scatter(x3,y5)
axis[4, 2].set_title('SUN - Very High Happiness')

# Combine all the operations and display
plt.show()


# In[67]:


# Scatter plots between all stations features and region features, for period 2014-15

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(5, 3)

# Determine the several x and y axis
x1 = h_w_one_year_df['TMAX']
x2 = h_w_one_year_df['RAIN']
x3 = h_w_one_year_df['SUN']
y1 = h_w_one_year_df['Average']
y2 = h_w_one_year_df['Low']
y3 = h_w_one_year_df['Medium']
y4 = h_w_one_year_df['High']
y5 = h_w_one_year_df['Very High']

axis[0, 0].scatter(x1,y1)
axis[0, 0].set_title('TMAX - Average Happiness')
axis[0, 1].scatter(x2,y1)
axis[0, 1].set_title('RAIN - Average Happiness')
axis[0, 2].scatter(x3,y1)
axis[0, 2].set_title('SUN - Average Happiness')
axis[1, 0].scatter(x1,y2)
axis[1, 0].set_title('TMAX - Low Happiness')
axis[1, 1].scatter(x2,y2)
axis[1, 1].set_title('RAIN - Low Happiness')
axis[1, 2].scatter(x3,y2)
axis[1, 2].set_title('SUN - Low Happiness')
axis[2, 0].scatter(x1,y3)
axis[2, 0].set_title('TMAX - Medium Happiness')
axis[2, 1].scatter(x2,y3)
axis[2, 1].set_title('RAIN - Medium Happiness')
axis[2, 2].scatter(x3,y3)
axis[2, 2].set_title('SUN - Medium Happiness')
axis[3, 0].scatter(x1,y4)
axis[3, 0].set_title('TMAX - High Happiness')
axis[3, 1].scatter(x2,y4)
axis[3, 1].set_title('RAIN - High Happiness')
axis[3, 2].scatter(x3,y4)
axis[3, 2].set_title('SUN - High Happiness')
axis[4, 0].scatter(x1,y5)
axis[4, 0].set_title('TMAX - Very High Happiness')
axis[4, 1].scatter(x2,y5)
axis[4, 1].set_title('RAIN - Very High Happiness')
axis[4, 2].scatter(x3,y5)
axis[4, 2].set_title('SUN - Very High Happiness')
  
# Combine all the operations and display
plt.show()


# In[ ]:




