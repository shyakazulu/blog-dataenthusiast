---
layout: post
title: TITANIC- Logistic Regression
---

###### Goal - Predict who surivies and who doesn't

![jpg](titanic_image.jpg)

- Source(https://bit.ly/35w815L)


```python
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
```


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
```


```python
%matplotlib inline
rcParams['figure.figsize'] = 5, 4
sb.set_style('whitegrid')
```

## Dataset


```python
tit = pd.read_csv('titanic-training-data.csv')
tit.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## FEATURES

Survived - Survival (0 = No; 1 = Yes)<br>
Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)<br>
Name - Name<br>
Sex - Sex<br>
Age - Age<br>
SibSp - Number of Siblings/Spouses Aboard<br>
Parch - Number of Parents/Children Aboard<br>
Ticket - Ticket Number<br>
Fare - Passenger Fare (British pound)<br>
Cabin - Cabin<br>
Embarked - Port of Embarkation (C = Cherbourg, France; Q = Queenstown, UK; S = Southampton - Cobh, Ireland)

### Is our target variable binary?


```python
sb.countplot(x='Survived', data=tit, palette ='hls')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x297a1761828>




![png](bar.png)


## Checking for missing values


```python
tit.isnull().any()
```




    PassengerId    False
    Survived       False
    Pclass         False
    Name           False
    Sex            False
    Age             True
    SibSp          False
    Parch          False
    Ticket         False
    Fare           False
    Cabin           True
    Embarked        True
    dtype: bool




```python
tit.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
tit.describe()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## Treating missing values


```python
tit.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')




```python
titanic_data=tit.drop(['Name','Ticket', 'Cabin'], axis=1)
titanic_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## Impute missing values

- we will use the Parch feature to give us a better mean estimate 


```python
sb.boxplot(x='Parch', y='Age', data=titanic_data, palette='hls')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x297a2bfa208>




![png](boxplot.png)



```python
parch_groups = titanic_data.groupby(titanic_data['Parch'])
parch_groups.mean()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Parch</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>445.255162</td>
      <td>0.343658</td>
      <td>2.321534</td>
      <td>32.178503</td>
      <td>0.237463</td>
      <td>25.586774</td>
    </tr>
    <tr>
      <th>1</th>
      <td>465.110169</td>
      <td>0.550847</td>
      <td>2.203390</td>
      <td>24.422000</td>
      <td>1.084746</td>
      <td>46.778180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>416.662500</td>
      <td>0.500000</td>
      <td>2.275000</td>
      <td>17.216912</td>
      <td>2.062500</td>
      <td>64.337604</td>
    </tr>
    <tr>
      <th>3</th>
      <td>579.200000</td>
      <td>0.600000</td>
      <td>2.600000</td>
      <td>33.200000</td>
      <td>1.000000</td>
      <td>25.951660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>384.000000</td>
      <td>0.000000</td>
      <td>2.500000</td>
      <td>44.500000</td>
      <td>0.750000</td>
      <td>84.968750</td>
    </tr>
    <tr>
      <th>5</th>
      <td>435.200000</td>
      <td>0.200000</td>
      <td>3.000000</td>
      <td>39.200000</td>
      <td>0.600000</td>
      <td>32.550000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>679.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>43.000000</td>
      <td>1.000000</td>
      <td>46.900000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def age_approx(cols):
    Age =cols[0]
    Parch=cols[0]
    
    if pd.isnull(Age):
        if Parch == 0:
            return 32
        elif Parch == 1:
            return 24
        elif Parch == 2:
            return 17
        elif Parch == 3:
            return 33
        elif Parch == 4:
            return 45
        else:
            return 30
    else:
        return Age
        
```


```python
titanic_data['Age']=titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
```


```python
titanic_data.isnull().sum()
```




    PassengerId    0
    Survived       0
    Pclass         0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Fare           0
    Embarked       2
    dtype: int64




```python
titanic_data.dropna(inplace=True)
titanic_data.reset_index(inplace=True, drop=True)
```


```python
print(titanic_data.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 889 entries, 0 to 888
    Data columns (total 9 columns):
    PassengerId    889 non-null int64
    Survived       889 non-null int64
    Pclass         889 non-null int64
    Sex            889 non-null object
    Age            889 non-null float64
    SibSp          889 non-null int64
    Parch          889 non-null int64
    Fare           889 non-null float64
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(2)
    memory usage: 62.6+ KB
    None
    

# Converting categorical variables to a dummy indicators


```python
from sklearn.preprocessing import LabelEncoder
label_encorder = LabelEncoder()
gender_cat = titanic_data['Sex']
gender_encoded =label_encorder .fit_transform(gender_cat)
gender_encoded[0:5]
```




    array([1, 0, 0, 0, 1])




```python
titanic_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



#### 1 = male   |   2 = female


```python
gender_DF = pd.DataFrame(gender_encoded, columns=['male_gender'])
gender_DF.head()
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
      <th>male_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
embarked_cat = titanic_data['Embarked']
embarked_encoded =label_encorder.fit_transform(embarked_cat)
embarked_encoded[0:20]
```




    array([2, 0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0])



#### This is a multi-nominal categorical variable and we need to one-hot encode it so we can have a binary output


```python
from sklearn.preprocessing import OneHotEncoder
binary_encoder = OneHotEncoder(categories='auto')
embarked_1hot =binary_encoder.fit_transform(embarked_encoded.reshape(-1,1))
embarked_1hot_mat = embarked_1hot.toarray()
embarked_DF=pd.DataFrame(embarked_1hot_mat, columns=['c','Q','S'])
embarked_DF.head()
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
      <th>c</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_data.drop(['Sex','Embarked'], axis=1, inplace=True)
titanic_data.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



#### concatenate the encoded variables


```python
titanic_dmy = pd.concat([titanic_data, embarked_DF,gender_DF], axis =1, verify_integrity=True).astype(float)
titanic_dmy[0:5] 
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>c</th>
      <th>Q</th>
      <th>S</th>
      <th>male_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Are the features correlated

- Logistic regression assumes that features are independent of one another
- correlation close to 1 or -1, implies a strong linear relationship between two variables.


```python
corr = titanic_dmy.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
```




<style  type="text/css" >
    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col0 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col1 {
            background-color:  #afcafc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col2 {
            background-color:  #a9c6fd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col3 {
            background-color:  #93b5fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col4 {
            background-color:  #6788ee;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col5 {
            background-color:  #7a9df8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col6 {
            background-color:  #b3cdfb;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col7 {
            background-color:  #ccd9ed;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col8 {
            background-color:  #a2c1ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col9 {
            background-color:  #cfdaea;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col10 {
            background-color:  #bad0f8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col0 {
            background-color:  #4961d2;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col1 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col2 {
            background-color:  #6687ed;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col3 {
            background-color:  #779af7;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col4 {
            background-color:  #6e90f2;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col5 {
            background-color:  #92b4fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col6 {
            background-color:  #e1dad6;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col7 {
            background-color:  #e5d8d1;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col8 {
            background-color:  #abc8fd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col9 {
            background-color:  #b1cbfc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col10 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col0 {
            background-color:  #4055c8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col1 {
            background-color:  #6485ec;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col2 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col3 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col4 {
            background-color:  #8fb1fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col5 {
            background-color:  #80a3fa;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col6 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col7 {
            background-color:  #9fbfff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col8 {
            background-color:  #d7dce3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col9 {
            background-color:  #d8dce2;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col10 {
            background-color:  #cbd8ee;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col0 {
            background-color:  #5470de;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col1 {
            background-color:  #9fbfff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col2 {
            background-color:  #6788ee;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col3 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col4 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col5 {
            background-color:  #4a63d3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col6 {
            background-color:  #c4d5f3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col7 {
            background-color:  #d2dbe8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col8 {
            background-color:  #a7c5fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col9 {
            background-color:  #c7d7f0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col10 {
            background-color:  #c3d5f4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col0 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col1 {
            background-color:  #a9c6fd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col2 {
            background-color:  #c3d5f4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col3 {
            background-color:  #506bda;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col4 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col5 {
            background-color:  #e4d9d2;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col6 {
            background-color:  #d2dbe8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col7 {
            background-color:  #c1d4f4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col8 {
            background-color:  #a3c2fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col9 {
            background-color:  #d7dce3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col10 {
            background-color:  #96b7ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col0 {
            background-color:  #4a63d3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col1 {
            background-color:  #c1d4f4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col2 {
            background-color:  #b5cdfa;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col3 {
            background-color:  #5d7ce6;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col4 {
            background-color:  #e3d9d3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col5 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col6 {
            background-color:  #dbdcde;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col7 {
            background-color:  #cad8ef;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col8 {
            background-color:  #97b8ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col9 {
            background-color:  #d6dce4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col10 {
            background-color:  #779af7;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col0 {
            background-color:  #4f69d9;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col1 {
            background-color:  #e1dad6;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col2 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col3 {
            background-color:  #a2c1ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col4 {
            background-color:  #a5c3fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col5 {
            background-color:  #b7cff9;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col6 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col7 {
            background-color:  #f1cdba;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col8 {
            background-color:  #8fb1fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col9 {
            background-color:  #aec9fc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col10 {
            background-color:  #88abfd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col0 {
            background-color:  #4a63d3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col1 {
            background-color:  #d3dbe7;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col2 {
            background-color:  #7a9df8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col3 {
            background-color:  #94b6ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col4 {
            background-color:  #6687ed;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col5 {
            background-color:  #779af7;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col6 {
            background-color:  #e4d9d2;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col7 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col8 {
            background-color:  #86a9fc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col9 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col10 {
            background-color:  #9dbdff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col0 {
            background-color:  #4055c8;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col1 {
            background-color:  #b1cbfc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col2 {
            background-color:  #dcdddd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col3 {
            background-color:  #89acfd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col4 {
            background-color:  #6f92f3;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col5 {
            background-color:  #6485ec;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col6 {
            background-color:  #97b8ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col7 {
            background-color:  #b2ccfb;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col8 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col9 {
            background-color:  #6c8ff1;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col10 {
            background-color:  #9fbfff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col0 {
            background-color:  #516ddb;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col1 {
            background-color:  #8db0fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col2 {
            background-color:  #c1d4f4;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col3 {
            background-color:  #85a8fc;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col4 {
            background-color:  #8badfd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col5 {
            background-color:  #8caffe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col6 {
            background-color:  #8caffe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col7 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col8 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col9 {
            background-color:  #b40426;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col10 {
            background-color:  #cad8ef;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col0 {
            background-color:  #5875e1;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col1 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col2 {
            background-color:  #cbd8ee;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col3 {
            background-color:  #a3c2fe;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col4 {
            background-color:  #5673e0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col5 {
            background-color:  #3b4cc0;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col6 {
            background-color:  #88abfd;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col7 {
            background-color:  #bed2f6;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col8 {
            background-color:  #98b9ff;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col9 {
            background-color:  #dedcdb;
        }    #T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col10 {
            background-color:  #b40426;
        }</style>  
<table id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2a" > 
<thead>    <tr> 
        <th class="blank level0" ></th> 
        <th class="col_heading level0 col0" >PassengerId</th> 
        <th class="col_heading level0 col1" >Survived</th> 
        <th class="col_heading level0 col2" >Pclass</th> 
        <th class="col_heading level0 col3" >Age</th> 
        <th class="col_heading level0 col4" >SibSp</th> 
        <th class="col_heading level0 col5" >Parch</th> 
        <th class="col_heading level0 col6" >Fare</th> 
        <th class="col_heading level0 col7" >c</th> 
        <th class="col_heading level0 col8" >Q</th> 
        <th class="col_heading level0 col9" >S</th> 
        <th class="col_heading level0 col10" >male_gender</th> 
    </tr></thead> 
<tbody>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row0" class="row_heading level0 row0" >PassengerId</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col0" class="data row0 col0" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col1" class="data row0 col1" >-0.005</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col2" class="data row0 col2" >-0.035</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col3" class="data row0 col3" >0.03</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col4" class="data row0 col4" >-0.058</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col5" class="data row0 col5" >-0.0017</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col6" class="data row0 col6" >0.013</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col7" class="data row0 col7" >-0.0012</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col8" class="data row0 col8" >-0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col9" class="data row0 col9" >0.022</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow0_col10" class="data row0 col10" >0.043</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row1" class="row_heading level0 row1" >Survived</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col0" class="data row1 col0" >-0.005</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col1" class="data row1 col1" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col2" class="data row1 col2" >-0.34</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col3" class="data row1 col3" >-0.076</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col4" class="data row1 col4" >-0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col5" class="data row1 col5" >0.083</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col6" class="data row1 col6" >0.26</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col7" class="data row1 col7" >0.17</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col8" class="data row1 col8" >0.0045</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col9" class="data row1 col9" >-0.15</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow1_col10" class="data row1 col10" >-0.54</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row2" class="row_heading level0 row2" >Pclass</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col0" class="data row2 col0" >-0.035</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col1" class="data row2 col1" >-0.34</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col2" class="data row2 col2" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col3" class="data row2 col3" >-0.33</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col4" class="data row2 col4" >0.082</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col5" class="data row2 col5" >0.017</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col6" class="data row2 col6" >-0.55</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col7" class="data row2 col7" >-0.25</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col8" class="data row2 col8" >0.22</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col9" class="data row2 col9" >0.076</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow2_col10" class="data row2 col10" >0.13</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row3" class="row_heading level0 row3" >Age</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col0" class="data row3 col0" >0.03</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col1" class="data row3 col1" >-0.076</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col2" class="data row3 col2" >-0.33</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col3" class="data row3 col3" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col4" class="data row3 col4" >-0.23</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col5" class="data row3 col5" >-0.18</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col6" class="data row3 col6" >0.088</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col7" class="data row3 col7" >0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col8" class="data row3 col8" >-0.0097</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col9" class="data row3 col9" >-0.024</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow3_col10" class="data row3 col10" >0.09</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row4" class="row_heading level0 row4" >SibSp</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col0" class="data row4 col0" >-0.058</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col1" class="data row4 col1" >-0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col2" class="data row4 col2" >0.082</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col3" class="data row4 col3" >-0.23</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col4" class="data row4 col4" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col5" class="data row4 col5" >0.41</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col6" class="data row4 col6" >0.16</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col7" class="data row4 col7" >-0.06</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col8" class="data row4 col8" >-0.027</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col9" class="data row4 col9" >0.069</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow4_col10" class="data row4 col10" >-0.12</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row5" class="row_heading level0 row5" >Parch</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col0" class="data row5 col0" >-0.0017</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col1" class="data row5 col1" >0.083</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col2" class="data row5 col2" >0.017</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col3" class="data row5 col3" >-0.18</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col4" class="data row5 col4" >0.41</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col5" class="data row5 col5" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col6" class="data row5 col6" >0.22</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col7" class="data row5 col7" >-0.012</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col8" class="data row5 col8" >-0.082</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col9" class="data row5 col9" >0.062</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow5_col10" class="data row5 col10" >-0.25</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row6" class="row_heading level0 row6" >Fare</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col0" class="data row6 col0" >0.013</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col1" class="data row6 col1" >0.26</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col2" class="data row6 col2" >-0.55</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col3" class="data row6 col3" >0.088</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col4" class="data row6 col4" >0.16</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col5" class="data row6 col5" >0.22</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col6" class="data row6 col6" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col7" class="data row6 col7" >0.27</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col8" class="data row6 col8" >-0.12</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col9" class="data row6 col9" >-0.16</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow6_col10" class="data row6 col10" >-0.18</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row7" class="row_heading level0 row7" >c</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col0" class="data row7 col0" >-0.0012</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col1" class="data row7 col1" >0.17</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col2" class="data row7 col2" >-0.25</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col3" class="data row7 col3" >0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col4" class="data row7 col4" >-0.06</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col5" class="data row7 col5" >-0.012</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col6" class="data row7 col6" >0.27</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col7" class="data row7 col7" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col8" class="data row7 col8" >-0.15</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col9" class="data row7 col9" >-0.78</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow7_col10" class="data row7 col10" >-0.085</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row8" class="row_heading level0 row8" >Q</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col0" class="data row8 col0" >-0.034</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col1" class="data row8 col1" >0.0045</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col2" class="data row8 col2" >0.22</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col3" class="data row8 col3" >-0.0097</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col4" class="data row8 col4" >-0.027</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col5" class="data row8 col5" >-0.082</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col6" class="data row8 col6" >-0.12</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col7" class="data row8 col7" >-0.15</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col8" class="data row8 col8" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col9" class="data row8 col9" >-0.5</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow8_col10" class="data row8 col10" >-0.075</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row9" class="row_heading level0 row9" >S</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col0" class="data row9 col0" >0.022</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col1" class="data row9 col1" >-0.15</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col2" class="data row9 col2" >0.076</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col3" class="data row9 col3" >-0.024</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col4" class="data row9 col4" >0.069</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col5" class="data row9 col5" >0.062</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col6" class="data row9 col6" >-0.16</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col7" class="data row9 col7" >-0.78</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col8" class="data row9 col8" >-0.5</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col9" class="data row9 col9" >1</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow9_col10" class="data row9 col10" >0.12</td> 
    </tr>    <tr> 
        <th id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2alevel0_row10" class="row_heading level0 row10" >male_gender</th> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col0" class="data row10 col0" >0.043</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col1" class="data row10 col1" >-0.54</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col2" class="data row10 col2" >0.13</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col3" class="data row10 col3" >0.09</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col4" class="data row10 col4" >-0.12</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col5" class="data row10 col5" >-0.25</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col6" class="data row10 col6" >-0.18</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col7" class="data row10 col7" >-0.085</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col8" class="data row10 col8" >-0.075</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col9" class="data row10 col9" >0.12</td> 
        <td id="T_3f92835c_2ff2_11ea_bc58_0019d2e12a2arow10_col10" class="data row10 col10" >1</td> 
    </tr></tbody> 
</table> 



##### fare and pclass are not independent of one another


```python
titanic_dmy.drop(['Fare','Pclass'], axis =1, inplace=True)
titanic_dmy.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>c</th>
      <th>Q</th>
      <th>S</th>
      <th>male_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Split data


```python
x_train,x_test,y_train,y_test = train_test_split(titanic_dmy.drop('Survived', axis=1),
                                                titanic_dmy['Survived'], test_size=0.2,
                                                random_state=200)
```


```python
print(x_train.shape)
print(y_train.shape)
```

    (711, 8)
    (711,)
    


```python
x_train[0:5]
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
      <th>PassengerId</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>c</th>
      <th>Q</th>
      <th>S</th>
      <th>male_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>719</th>
      <td>721.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>165</th>
      <td>167.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>879</th>
      <td>882.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>451</th>
      <td>453.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>181</th>
      <td>183.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



## Model creation


```python
LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(x_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='liblinear',
              tol=0.0001, verbose=0, warm_start=False)



#### use the model to make a prediction


```python
y_pred =LogReg.predict(x_test)
```

## Model Evaluation
### Classification report without cross-validation


```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
             0.0       0.83      0.88      0.85       109
             1.0       0.79      0.71      0.75        69
    
       micro avg       0.81      0.81      0.81       178
       macro avg       0.81      0.80      0.80       178
    weighted avg       0.81      0.81      0.81       178
    
    

### K-fold cross-validation & confusion matrices


```python
y_train_pred =cross_val_predict(LogReg, x_train, y_train, cv=5)
confusion_matrix(y_train, y_train_pred)
```




    array([[377,  63],
           [ 91, 180]], dtype=int64)



##### correct predictions = 377 and 180  | incorrect predictions=91 and 63


```python
precision_score(y_train, y_train_pred)
```




    0.7407407407407407



### Test prediction


```python
titanic_dmy[52:53]
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>c</th>
      <th>Q</th>
      <th>S</th>
      <th>male_gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52</th>
      <td>53.0</td>
      <td>1.0</td>
      <td>49.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_passenger =np.array([53,42,1,0,1,0,0,0]).reshape(1,-1)
print(LogReg.predict(test_passenger))
print(LogReg.predict_proba(test_passenger))
```

    [1.]
    [[0.22282159 0.77717841]]
    

### our model predicts that our test_passenger with a different age will survive with an approximate precission of 78%


