---
layout : jupyter
title : pandas.DataFrame.any(), numpy.any()
category : Code Snippet
tags : pandas numpy
---
***

평소에 헷갈리는 any(), all()에 대해 정리하였습니다. 

***

<h1>df.isna()</h1>

<div class="prompt input_prompt">
In&nbsp;[1]:
</div>

```python
import pandas as pd
```

<div class="prompt input_prompt">
In&nbsp;[2]:
</div>

```python
df = pd.read_csv('./data/top1_1880109251922.csv', index_col=[0])
df['date'] = pd.to_datetime(df['date'])
df['Date'] = pd.to_datetime(df['date'])
df = df.set_index('Date')
df = df.asfreq('D')
```

df.isna()는 데이터프레임에서 NaN 요소에 해당되는 부분을 True로 리턴해준다.

<div class="prompt input_prompt">
In&nbsp;[3]:
</div>

```python
df.isna()
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
      <th>date</th>
      <th>store</th>
      <th>product_c</th>
      <th>sales</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-02-01</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2018-02-02</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2018-02-03</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2018-02-04</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2018-02-05</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-07-27</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2019-07-28</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2019-07-29</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2019-07-30</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2019-07-31</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>546 rows × 4 columns</p>
</div>




<h1>df.any()</h1>

여기서, dataframe.any(axis=0)인 경우엔 각 column의 row를 다 훑어서, row요소들 중 적어도 하나의 row애 True가 있으면, True를 반환합니다.

<div class="prompt input_prompt">
In&nbsp;[4]:
</div>

```python
df.isna().any(axis=0)
```





{:.output_data_text}
    date         True
    store        True
    product_c    True
    sales        True
    dtype: bool




반면에, dataframe.any(axis=1)인 경우엔 각 index별로 column요소를 다 훑어서 적어도 하나의 column에 True가 있으면 True를 반환합니다.
아래 코드를 보면, 해당 index에 data, store, product_c, sales가 모두 True이면 해당 index row는 True를 반환합니다.

<div class="prompt input_prompt">
In&nbsp;[5]:
</div>

```python
df.isna().any(axis=1)
```





{:.output_data_text}
    Date
    2018-02-01    False
    2018-02-02    False
    2018-02-03    False
    2018-02-04    False
    2018-02-05    False
                  ...  
    2019-07-27     True
    2019-07-28     True
    2019-07-29    False
    2019-07-30    False
    2019-07-31    False
    Freq: D, Length: 546, dtype: bool




<h1>np.any()</h1>

np.any() 는 dataframe.any()와 유사합니다. 주어진 축(axis) 정보에 따라 해당 요소에서 True가 하나 이상이라도 있으면 True를 반환합니다. 아래는 np.any()의 예제입니다.

<div class="prompt input_prompt">
In&nbsp;[6]:
</div>

```python
import numpy as np
```

먼저, 예제 array를 생성합니다. True, False로 구성된 random한 array를 만들었습니다.

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

```python
samplearr = [True, False]
a = np.random.choice(samplearr, size=(47))
b = np.random.choice(samplearr, size=(47))
c = np.random.choice(samplearr, size=(47))
print(a)
print(b)
print(c)
```

{:.output_stream}
    [ True  True  True  True  True  True False  True  True  True False False
     False  True False  True False False False False  True False  True  True
     False  True False False False  True False  True False  True  True False
     False  True False False  True False False  True  True  True False]
    [False False  True False  True False  True  True False  True  True False
      True False  True False False False False False  True  True False False
     False False  True False False  True False False False  True False  True
      True  True False False  True  True  True  True False  True  True]
    [False False  True False False False False  True False False  True  True
      True False False  True  True  True  True False False False False  True
     False  True  True  True  True  True  True  True False False False  True
     False False False False False False  True  True  True  True  True]


np.any(a,b,c)는 에러를 발생합니다. 반드시, 하나의 array나 아니면 array와 유사한 list형식으로 묶어서 넣어줘야합니다.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

```python
np.any(a,b,c)
```



{:.output_traceback_line}
    ---------------------------------------------------------------------------

{:.output_traceback_line}
    TypeError                                 Traceback (most recent call last)

{:.output_traceback_line}
    <ipython-input-8-7a7facd3228c> in <module>
    ----> 1 np.any(a,b,c)
    

{:.output_traceback_line}
    <__array_function__ internals> in any(*args, **kwargs)


{:.output_traceback_line}
    /opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py in any(a, axis, out, keepdims)
       2328 
       2329     """
    -> 2330     return _wrapreduction(a, np.logical_or, 'any', axis, None, out, keepdims=keepdims)
       2331 
       2332 


{:.output_traceback_line}
    /opt/anaconda3/lib/python3.7/site-packages/numpy/core/fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
         85                 return reduction(axis=axis, out=out, **passkwargs)
         86 
    ---> 87     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
         88 
         89 


{:.output_traceback_line}
    TypeError: only integer scalar arrays can be converted to a scalar index



<h2>axis=0 vs. axis=1</h2>

그전에 axis=0과 1에 따라 차이를 살펴봅시다. axis=0인 경우엔 각 column의 모든 row를 훑고, axis=1인 경우엔 각 row의 모든 column을 훑습니다. 아래는 관련 그림입니다.

![jpg](/images/2020-12-01-any-all-usage_files/axis.jpg)

axis=0인 경우, 각각의 column요소에서 모든 row를 훑어서 하나 이상이 True요소라면 True를 반환합니다. 결과는 [a,b,c]의 column의 갯수만큼 출력됩니다.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

```python
np.any([a,b,c], axis=0)
```





{:.output_data_text}
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True, False,  True,  True,  True,  True, False,  True,  True,
            True,  True,  True,  True,  True, False,  True,  True,  True,
            True,  True, False, False,  True,  True,  True,  True,  True,
            True,  True])




axis=1인 경우, 각각의 row요소에서 모든 column을 훑어서 하나 이상이 True요소라면 True를 반환합니다. 결과는 [a,b,c]의 row 갯수만큼 출력됩니다.

<div class="prompt input_prompt">
In&nbsp;[10]:
</div>

```python
np.any([a,b,c], axis=1)
```





{:.output_data_text}
    array([ True,  True,  True])




<h1>np.all()</h1>

np.all()은 np.any()와 반대로, 검사할 축에 모든 요소가 True여야지만 True를 반환합니다. 아래는 예제입니다.

<div class="prompt input_prompt">
In&nbsp;[11]:
</div>

```python
np.all([a,b,c], axis=0)
```





{:.output_data_text}
    array([False, False,  True, False, False, False, False,  True, False,
           False, False, False, False, False, False, False, False, False,
           False, False, False, False, False, False, False, False, False,
           False, False,  True, False, False, False, False, False, False,
           False, False, False, False, False, False, False,  True, False,
            True, False])




<div class="prompt input_prompt">
In&nbsp;[12]:
</div>

```python
np.all([a,b,c], axis=1)
```





{:.output_data_text}
    array([False, False, False])



