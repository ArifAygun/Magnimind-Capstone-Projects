# Data Experimentation Capstone

## Introduction

The objective of this project is to examine the effects of user engagement after seeing Ad A or Ad B

## Problem

An advertising company has developed a new ad to have users engage with their questionnaire. The company has shown the new ad to some users and a dummy ad to others and wants their data analyst team to interpret the results. Does the new ad generate more responses to their questionnaire? Is it statistically significant? Is the company justified in using the new ad? 

A/B testing is common in the business world and is a way to compare two versions of something to figure out which performs better. Figuring out which ad users prefer is a real life business problem that would be expected to know how to solve as a business data analyst. 


## Data

Dataset is found on kaggle from an advertising company. https://www.kaggle.com/osuolaleemmanuel/ad-ab-testing

## Importing Libraries


```python
import pandas as pd
import numpy as np
```

## Reading The Data


```python
# setting dataframe
df = pd.read_csv("AdSmartABdata.csv")
```


```python
# checking head
df.head()
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
      <th>auction_id</th>
      <th>experiment</th>
      <th>date</th>
      <th>hour</th>
      <th>device_make</th>
      <th>platform_os</th>
      <th>browser</th>
      <th>yes</th>
      <th>no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0008ef63-77a7-448b-bd1e-075f42c55e39</td>
      <td>exposed</td>
      <td>2020-07-10</td>
      <td>8</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000eabc5-17ce-4137-8efe-44734d914446</td>
      <td>exposed</td>
      <td>2020-07-07</td>
      <td>10</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0016d14a-ae18-4a02-a204-6ba53b52f2ed</td>
      <td>exposed</td>
      <td>2020-07-05</td>
      <td>2</td>
      <td>E5823</td>
      <td>6</td>
      <td>Chrome Mobile WebView</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00187412-2932-4542-a8ef-3633901c98d9</td>
      <td>control</td>
      <td>2020-07-03</td>
      <td>15</td>
      <td>Samsung SM-A705FN</td>
      <td>6</td>
      <td>Facebook</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>001a7785-d3fe-4e11-a344-c8735acacc2c</td>
      <td>control</td>
      <td>2020-07-03</td>
      <td>15</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8077 entries, 0 to 8076
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   auction_id   8077 non-null   object
     1   experiment   8077 non-null   object
     2   date         8077 non-null   object
     3   hour         8077 non-null   int64 
     4   device_make  8077 non-null   object
     5   platform_os  8077 non-null   int64 
     6   browser      8077 non-null   object
     7   yes          8077 non-null   int64 
     8   no           8077 non-null   int64 
    dtypes: int64(4), object(5)
    memory usage: 568.0+ KB


There are 8077 rows and 9 columns in the dataset.

## The Data Wrangling

The dataset contains the yes and no columns. A 1 in one the columns indicates it was selected, and a 0 indicates it wasn't.
A 0 in both indicates the user ignored the questionnaire.
Users who answered and whether they answered yes or no is most important therefore all rows with 0 in both yes/no columns will be dropped


```python
drop = df.loc[(df["yes"]==0) & (df["no"]==0)]
drop.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6834 entries, 0 to 8076
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   auction_id   6834 non-null   object
     1   experiment   6834 non-null   object
     2   date         6834 non-null   object
     3   hour         6834 non-null   int64 
     4   device_make  6834 non-null   object
     5   platform_os  6834 non-null   int64 
     6   browser      6834 non-null   object
     7   yes          6834 non-null   int64 
     8   no           6834 non-null   int64 
    dtypes: int64(4), object(5)
    memory usage: 533.9+ KB


6834 users in the dataset did not answer the questioner so they will be dropped.


```python
df.drop(drop.index, axis=0,inplace=True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1243 entries, 2 to 8071
    Data columns (total 9 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   auction_id   1243 non-null   object
     1   experiment   1243 non-null   object
     2   date         1243 non-null   object
     3   hour         1243 non-null   int64 
     4   device_make  1243 non-null   object
     5   platform_os  1243 non-null   int64 
     6   browser      1243 non-null   object
     7   yes          1243 non-null   int64 
     8   no           1243 non-null   int64 
    dtypes: int64(4), object(5)
    memory usage: 97.1+ KB


After dropping users who did not answer we have 1243 enteries left.

We will now create a new column "answer" where 0 indicates no and 1 indicates yes then drop the yes & no columns


```python
# creating answer column
df["answer"] = df["yes"]

# dropping yes and no columns
df.drop(["yes","no"], axis=1, inplace=True)

# checking dataframe
df.head()
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
      <th>auction_id</th>
      <th>experiment</th>
      <th>date</th>
      <th>hour</th>
      <th>device_make</th>
      <th>platform_os</th>
      <th>browser</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0016d14a-ae18-4a02-a204-6ba53b52f2ed</td>
      <td>exposed</td>
      <td>2020-07-05</td>
      <td>2</td>
      <td>E5823</td>
      <td>6</td>
      <td>Chrome Mobile WebView</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>008aafdf-deef-4482-8fec-d98e3da054da</td>
      <td>exposed</td>
      <td>2020-07-04</td>
      <td>16</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>00a1384a-5118-4d1b-925b-6cdada50318d</td>
      <td>exposed</td>
      <td>2020-07-06</td>
      <td>8</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>00b6fadb-10bd-49e3-a778-290da82f7a8d</td>
      <td>control</td>
      <td>2020-07-08</td>
      <td>4</td>
      <td>Samsung SM-A202F</td>
      <td>6</td>
      <td>Facebook</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>00ebf4a8-060f-4b99-93ac-c62724399483</td>
      <td>control</td>
      <td>2020-07-03</td>
      <td>15</td>
      <td>Generic Smartphone</td>
      <td>6</td>
      <td>Chrome Mobile</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking datatypes
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1243 entries, 2 to 8071
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   auction_id   1243 non-null   object
     1   experiment   1243 non-null   object
     2   date         1243 non-null   object
     3   hour         1243 non-null   int64 
     4   device_make  1243 non-null   object
     5   platform_os  1243 non-null   int64 
     6   browser      1243 non-null   object
     7   answer       1243 non-null   int64 
    dtypes: int64(3), object(5)
    memory usage: 87.4+ KB


## Exploratory Data Analysis


```python
# converting date column to datetime
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1243 entries, 2 to 8071
    Data columns (total 8 columns):
     #   Column       Non-Null Count  Dtype         
    ---  ------       --------------  -----         
     0   auction_id   1243 non-null   object        
     1   experiment   1243 non-null   object        
     2   date         1243 non-null   datetime64[ns]
     3   hour         1243 non-null   int64         
     4   device_make  1243 non-null   object        
     5   platform_os  1243 non-null   int64         
     6   browser      1243 non-null   object        
     7   answer       1243 non-null   int64         
    dtypes: datetime64[ns](1), int64(3), object(4)
    memory usage: 87.4+ KB



```python
# checking count of unique values in experiment column
df.groupby('experiment')[['auction_id']].count()
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>experiment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>control</th>
      <td>586</td>
    </tr>
    <tr>
      <th>exposed</th>
      <td>657</td>
    </tr>
  </tbody>
</table>
</div>



There are 586 users in the control group and 657 users in the exposed group.


```python
# checking count of unique values in date column
df_date=df.groupby('date')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_date
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-03</th>
      <td>325</td>
    </tr>
    <tr>
      <th>2020-07-09</th>
      <td>184</td>
    </tr>
    <tr>
      <th>2020-07-08</th>
      <td>177</td>
    </tr>
    <tr>
      <th>2020-07-04</th>
      <td>159</td>
    </tr>
    <tr>
      <th>2020-07-10</th>
      <td>124</td>
    </tr>
    <tr>
      <th>2020-07-05</th>
      <td>117</td>
    </tr>
    <tr>
      <th>2020-07-07</th>
      <td>83</td>
    </tr>
    <tr>
      <th>2020-07-06</th>
      <td>74</td>
    </tr>
  </tbody>
</table>
</div>



Most replies were made on July 3rd


```python
# checking count of unique values in hour column
df_hour = df.groupby('hour')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_hour
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>hour</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>281</td>
    </tr>
    <tr>
      <th>8</th>
      <td>67</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59</td>
    </tr>
    <tr>
      <th>10</th>
      <td>57</td>
    </tr>
    <tr>
      <th>9</th>
      <td>57</td>
    </tr>
    <tr>
      <th>14</th>
      <td>51</td>
    </tr>
    <tr>
      <th>6</th>
      <td>50</td>
    </tr>
    <tr>
      <th>5</th>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>46</td>
    </tr>
    <tr>
      <th>20</th>
      <td>45</td>
    </tr>
    <tr>
      <th>11</th>
      <td>44</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44</td>
    </tr>
    <tr>
      <th>16</th>
      <td>44</td>
    </tr>
    <tr>
      <th>12</th>
      <td>42</td>
    </tr>
    <tr>
      <th>13</th>
      <td>41</td>
    </tr>
    <tr>
      <th>18</th>
      <td>38</td>
    </tr>
    <tr>
      <th>19</th>
      <td>38</td>
    </tr>
    <tr>
      <th>21</th>
      <td>34</td>
    </tr>
    <tr>
      <th>17</th>
      <td>34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>32</td>
    </tr>
    <tr>
      <th>0</th>
      <td>31</td>
    </tr>
    <tr>
      <th>22</th>
      <td>21</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



The users are most active from early morning hours to afternoon hours (6-11 to 12-16).
And least active at night (19-2).


```python
# checking count of unique values in devive maker column
df_device = df.groupby('device_make')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_device
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>device_make</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Generic Smartphone</th>
      <td>719</td>
    </tr>
    <tr>
      <th>Samsung SM-G960F</th>
      <td>45</td>
    </tr>
    <tr>
      <th>Samsung SM-G950F</th>
      <td>35</td>
    </tr>
    <tr>
      <th>Samsung SM-G973F</th>
      <td>22</td>
    </tr>
    <tr>
      <th>Samsung SM-A202F</th>
      <td>18</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>Samsung SM-G925F</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Samsung SM-G965U1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Samsung SM-J330F</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Samsung SM-J330G</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Samsung SM-A750GN</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 1 columns</p>
</div>



Most of the users had a Generic Smartphone or a Samsung phone.


```python
# checking count of unique values in platform_os column
df_os=df.groupby('platform_os')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_os
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>platform_os</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>1226</td>
    </tr>
    <tr>
      <th>5</th>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>



Majority of users were on Platform Os 6


```python
# checking count of unique values in browser column
df_browser=df.groupby('browser')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_browser
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>browser</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chrome Mobile</th>
      <td>695</td>
    </tr>
    <tr>
      <th>Chrome Mobile WebView</th>
      <td>227</td>
    </tr>
    <tr>
      <th>Facebook</th>
      <td>156</td>
    </tr>
    <tr>
      <th>Samsung Internet</th>
      <td>145</td>
    </tr>
    <tr>
      <th>Mobile Safari</th>
      <td>14</td>
    </tr>
    <tr>
      <th>Mobile Safari UI/WKWebView</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Chrome</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Chrome Mobile iOS</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The most used browsers when ansering the questionnaire were Chrome Mobile, Chrome Mobile WebView, Facebook and Samsung Internet.


```python
# checking count of unique values in answer column
df_answer=df.groupby('answer')[['auction_id']].count().sort_values(by='auction_id', ascending=False)
df_answer
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
      <th>auction_id</th>
    </tr>
    <tr>
      <th>answer</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>671</td>
    </tr>
    <tr>
      <th>1</th>
      <td>572</td>
    </tr>
  </tbody>
</table>
</div>



There was overall more No (0) than Yes(1) in the replies.

## Statistical Analysis


```python
# selecting a sample of the dataset for equal comparison
required_n = 586
control_sample = df[df['experiment'] == 'control'].sample(n=required_n, random_state=22)
exposed_sample = df[df['experiment'] == 'exposed'].sample(n=required_n, random_state=22)

ab_test = pd.concat([control_sample, exposed_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
```

## Comparing statistics of groups


```python
import scipy.stats as stats
conversion_rates = ab_test.groupby('experiment')['answer']

std_p = lambda x: np.std(x, ddof=0)              # Std. deviation of the proportion
se_p = lambda x: stats.sem(x, ddof=0)            # Std. error of the proportion (std / sqrt(n))

conversion_rates = conversion_rates.agg([np.mean, std_p, se_p])
conversion_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']


conversion_rates.style.format('{:.3f}')
```




<style type="text/css">
</style>
<table id="T_927e5">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_927e5_level0_col0" class="col_heading level0 col0" >conversion_rate</th>
      <th id="T_927e5_level0_col1" class="col_heading level0 col1" >std_deviation</th>
      <th id="T_927e5_level0_col2" class="col_heading level0 col2" >std_error</th>
    </tr>
    <tr>
      <th class="index_name level0" >experiment</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_927e5_level0_row0" class="row_heading level0 row0" >control</th>
      <td id="T_927e5_row0_col0" class="data row0 col0" >0.451</td>
      <td id="T_927e5_row0_col1" class="data row0 col1" >0.498</td>
      <td id="T_927e5_row0_col2" class="data row0 col2" >0.021</td>
    </tr>
    <tr>
      <th id="T_927e5_level0_row1" class="row_heading level0 row1" >exposed</th>
      <td id="T_927e5_row1_col0" class="data row1 col0" >0.457</td>
      <td id="T_927e5_row1_col1" class="data row1 col1" >0.498</td>
      <td id="T_927e5_row1_col2" class="data row1 col2" >0.021</td>
    </tr>
  </tbody>
</table>





Judging by the stats above, it does look like our exposed group design performed similarly, with our new design performing slightly better, at 45.1% vs. 45.7% conversion rate.

## Hypothesis Testing


```python
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
```


```python
control_results = ab_test[ab_test['experiment'] == 'control']['answer']
exposed_results = ab_test[ab_test['experiment'] == 'exposed']['answer']
```


```python
n_con = control_results.count()
n_treat = exposed_results.count()
successes = [control_results.sum(), exposed_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for control group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for treatment group: [{lower_treat:.3f}, {upper_treat:.3f}]')
```

    z statistic: -0.23
    p-value: 0.814
    ci 95% for control group: [0.410, 0.491]
    ci 95% for treatment group: [0.417, 0.498]


## Conclusion

- Since our $p$-value=0.814 is far above our $\alpha$=0.05, we cannot reject the null hypothesis $H_0$, which means that the new advertisement design did not performed better than the old one.

- There were enough data points to make a reasonable judgement.

- Finally, based on the A/B Testing Analysis the new advertisement design does not give an increase in brand awareness.

## References

- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html

- https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html

- https://www.yourdatateacher.com/2022/10/17/a-beginners-guide-to-statistical-hypothesis-tests/
