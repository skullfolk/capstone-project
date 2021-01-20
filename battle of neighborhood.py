#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd
import folium
import requests
import numpy as np
from pandas.io.json import json_normalize
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns


# In[3]:


url='https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
wiki=requests.get(url)
df=pd.read_html(wiki.content)[0]
df


# In[82]:


df.drop(df.loc[df['Borough']=='Not assigned'].index,inplace=True)
df


# In[5]:


df1=df.groupby('Postal Code')['Neighbourhood'].apply(lambda x:'%s'% ','.join(x))
df1=df1.to_frame()
df1=df1.reset_index()
#df1.rename(column={'Neighbourhood': "new_Neighbourhood"},inplace=True)
df1


# In[20]:


df2=pd.merge(df1,df,on='Postal Code')
df2.drop('Neighbourhood_y',axis=1,inplace=True)
df2


# In[21]:


coor=pd.read_csv('http://cocl.us/Geospatial_data')
coor.head()


# In[22]:


df2=pd.merge(coor,df2,on='Postal Code')
df2=df2[['Postal Code','Borough','Neighbourhood_x','Latitude','Longitude']]
df2


# In[23]:


CLIENT_ID='UPZOGDSSYOESSLI03TJJRMMLXJA5JBRYUDWKH2TDSAOJ5RN2'
CLIENT_SECRET='4OIK45JO4PTXAXUBHUYZHXMHQYFZU31MX1VIZ3ACDDZIVG45'
VERSION='20180605'
radius=500
LIMIT=100


# In[24]:


df2=df2[df2['Borough'].str.contains('Toronto')]
df2


# In[25]:


Latitude= 43.651070
Longitude= -79.347015
map_toronto=folium.Map(location=[Latitude,Longitude],zoom_start=10)
for lat,lon,borough,neighbourhood in zip(df2['Latitude'],df2['Longitude'],df2['Borough'],df2['Neighbourhood_x']):
    labels='{}, {}'.format(neighbourhood, borough)
    labels = folium.Popup(labels,parse_html=True)
    folium.CircleMarker(
    [lat,lon],
    radius=5,
    popup=labels,
    color='blue',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.7,
    parse_html=False).add_to(map_toronto)
map_toronto


# In[26]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    Latitude, 
    Longitude, 
    radius, 
    LIMIT)
url


# In[27]:


result=requests.get(url).json()
result


# In[29]:


def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[30]:


venues=result['response']['groups'][0]['items']
venues


# In[31]:


nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues


# In[32]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[35]:


toronto_venues = getNearbyVenues(names=df2['Neighbourhood_x'],
                                   latitudes=df2['Latitude'],
                                   longitudes=df2['Longitude'])
                                  


# In[36]:


toronto_venues


# In[37]:


toronto_onehot=pd.get_dummies(toronto_venues['Venue Category'])
toronto_onehot


# In[38]:


res_cols = [col for col in toronto_onehot.columns if 'Restaurant' in col]
res_cols


# In[46]:


toronto_onehot=toronto_onehot[res_cols]
toronto_onehot


# In[100]:


toronto_onehot['Neighborhood'] = toronto_venues['Neighborhood'] 
toronto_onehot


# In[101]:


fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]


# In[102]:


toronto_onehot.head()


# In[104]:


toronto_group=toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_group


# In[105]:


kcluster=5
toronto_cluster=toronto_group.drop('Neighborhood',axis=1)
kmean=KMeans(n_clusters=kcluster,random_state=0).fit(toronto_cluster)
kmean.labels_[0:10]


# In[108]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in range(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = toronto_grouped['Neighborhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[ ]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmean.labels_)


# In[115]:


neighborhoods_venues_sorted


# In[122]:


toronto_merged = df2.drop('Postal Code',axis=1)
toronto_merged = pd.merge(toronto_merged,neighborhoods_venues_sorted,on='Neighborhood')
toronto_merged


# In[123]:


map_clusters = folium.Map(location=[Latitude, Longitude], zoom_start=11)

x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighborhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[125]:


def plot_bar(clusternumber):
    sns.set(style="whitegrid",rc={'figure.figsize':(20,10)})
    df = clusterdata[[clusternumber]].drop(clusterdata[[clusternumber]][clusterdata[clusternumber]==0].index)
    chart = sns.barplot(x=df.index, y=clusternumber, data=df)
    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)


# In[126]:


clusterdata = pd.merge(toronto_onehot.groupby('Neighborhood').sum(),toronto_merged[['Neighborhood','Cluster Labels']],left_on='Neighborhood', right_on='Neighborhood',how='inner')
clusterdata = clusterdata.iloc[:,1:].groupby('Cluster Labels').sum().transpose()
clusterdata.head()


# In[127]:


plot_bar(0)


# In[129]:


plot_bar(1)


# In[130]:


plot_bar(2)


# In[131]:


plot_bar(3)


# In[132]:


plot_bar(4)


# In[144]:


clusterdata[0].sum()


# In[149]:


toronto_merged2=toronto_merged.drop(toronto_merged.loc[toronto_merged['Cluster Labels']!=0].index)
toronto_merged2


# In[151]:


map_toronto=folium.Map(location=[Latitude,Longitude],zoom_start=10)
for lat,lon,borough,neighbourhood in zip(toronto_merged2['Latitude'],toronto_merged2['Longitude'],toronto_merged2['Borough'],toronto_merged2['Neighborhood']):
    labels='{}, {}'.format(neighbourhood, borough)
    labels = folium.Popup(labels,parse_html=True)
    folium.CircleMarker(
    [lat,lon],
    radius=5,
    popup=labels,
    color='blue',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.7,
    parse_html=False).add_to(map_toronto)
map_toronto


# In[ ]:




