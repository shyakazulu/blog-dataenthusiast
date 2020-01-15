---
layout: post
title: Visualizing Bird Migration
---

```python
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
import matplotlib.pyplot as plt 
import pandas as pd
```

#### Load the dataset. you can get the dataset from [LifeWatch INBO project](https://inbo.carto.com/u/lifewatch/datasets).


```python
birds = pd.read_csv('tracking.csv')
birds.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>altitude</th>
      <th>date_time</th>
      <th>device_info_serial</th>
      <th>direction</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>speed_2d</th>
      <th>bird_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>71</td>
      <td>2013-08-15 00:18:08+00</td>
      <td>851</td>
      <td>-150.469753</td>
      <td>49.419859</td>
      <td>2.120733</td>
      <td>0.150000</td>
      <td>Eric</td>
    </tr>
    <tr>
      <td>1</td>
      <td>68</td>
      <td>2013-08-15 00:48:07+00</td>
      <td>851</td>
      <td>-136.151141</td>
      <td>49.419880</td>
      <td>2.120746</td>
      <td>2.438360</td>
      <td>Eric</td>
    </tr>
    <tr>
      <td>2</td>
      <td>68</td>
      <td>2013-08-15 01:17:58+00</td>
      <td>851</td>
      <td>160.797477</td>
      <td>49.420310</td>
      <td>2.120885</td>
      <td>0.596657</td>
      <td>Eric</td>
    </tr>
    <tr>
      <td>3</td>
      <td>73</td>
      <td>2013-08-15 01:47:51+00</td>
      <td>851</td>
      <td>32.769360</td>
      <td>49.420359</td>
      <td>2.120859</td>
      <td>0.310161</td>
      <td>Eric</td>
    </tr>
    <tr>
      <td>4</td>
      <td>69</td>
      <td>2013-08-15 02:17:42+00</td>
      <td>851</td>
      <td>45.191230</td>
      <td>49.420331</td>
      <td>2.120887</td>
      <td>0.193132</td>
      <td>Eric</td>
    </tr>
  </tbody>
</table>
</div>



#### Let's look at the name of birds we are tracking.a


```python
bird_names = pd.unique(birds.bird_name)  
bird_names
```




    array(['Eric', 'Nico', 'Sanne'], dtype=object)



### Plot the migration pattern using longitude and latitude featues of 'Eric' 


```python
select_name = birds.bird_name == "Eric" 
x,y = birds.longitude[select_name], birds.latitude[select_name] 
plt.figure(figsize = (7,7)) 
plt.plot(x,y,"b.") 
```

![image](/assets/images/one_bird.png)



### Plot the migration pattern of the three birds in our sample dataset.


```python
plt.figure(figsize = (7,7)) 
for bird_name in bird_names: 
    # storing the indices of the bird Eric 
    select_name = birds.bird_name == bird_name   
    x,y = birds.longitude[select_name], birds.latitude[select_name] 
    plt.plot(x,y,".", label=bird_name) 
plt.xlabel("Longitude") 
plt.ylabel("Latitude") 
plt.legend(loc="lower right") 
plt.show() 
```


![image](/assets/images/two_bird.png)


### Let's add a map to enrich thr birds migration narrative.


```python
proj = ccrs.Mercator()  
  
plt.figure(figsize=(10,10)) 
ax = plt.axes(projection=proj) 
ax.set_extent((-25.0, 20.0, 52.0, 10.0)) 
ax.add_feature(cfeature.LAND) 
ax.add_feature(cfeature.OCEAN) 
ax.add_feature(cfeature.COASTLINE) 
ax.add_feature(cfeature.BORDERS, linestyle=':') 
for name in bird_names: 
    select_name = birds['bird_name'] == name 
    x,y = birds.longitude[select_name], birds.latitude[select_name] 
    ax.plot(x,y,'.', transform=ccrs.Geodetic(), label=name) 
plt.legend(loc="upper left") 
plt.show() 
```


![image](/assets/images/mapmigration.png)



```python

```
