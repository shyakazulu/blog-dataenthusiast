---

layout: post

title: Exploratory Data Analysis using Python_Profiler

---

- EDA plays a crucial role in understanding data.
- Provides more insight into the data. For instance, correlation, mean and other measures of central tendency.
- Pandas profiling reduces the amount of code that we have to write in order to get a feel for the data.
- With just one line of code, we get the following statistics;
      - Essentials: type, unique values, missing values
      - Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
      - Descriptive statistics like mean, mode, standard deviation, coefficient of variation, kurtosis, skewness
      - Most frequent values
      - Histogram
      - Correlations highlighting of highly correlated variables, Spearman and Pearson matrixes      


```python
import pandas as pd
import pandas_profiling as pp
from pandas_profiling import ProfileReport
```


```python
dataset =pd.read_csv('titanic.csv')
```

###### Generate the profile report 

```python
profilerep = ProfileReport(dataset, title="Titanic Dataset", html={'style': {'full_width': True}})
profilerep
```

![image](/assets/images/overview.JPG)

![image](/assets/images/features.JPG)

![image](/assets/images/correlations.JPG)

![image](/assets/images/missing.JPG)

![image](/assets/images/sample.JPG)



###### use the code below to generate a HTML report file...


```python
profilerep.to_file(output_file="titanic_profile.html")
```

#### Reerence Link:     [https://pypi.org/project/pandas-profiling/]


```python

```

