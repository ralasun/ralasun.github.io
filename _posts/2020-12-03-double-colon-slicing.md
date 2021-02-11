---
layout : jupyter
title : dataframe, numpy 등 array에서 double-colon(::) slicing 
category : Code Snippet
tags : pandas
---
---

pandas, numpy 등 자주 헷갈리는 코드 사용을 모아두었습니다.

---

# df[::c]

시작부터 c 간격마다 있는 row를 슬라이싱해줍니다. 자세히 설명하면, 1번째, (1+c)번째, (1+2c)번째, ..., (1+nc)번째 row가 선택됩니다. 아래는 예제입니다.


```python
import pandas as pd
import numpy as np
```


```python
a = np.random.normal(size=200)
b = np.random.uniform(size=200)
sampledf = pd.DataFrame({'A':a,'B':b})
```


```python
sampledf[::2]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.312234</td>
      <td>0.788584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.123720</td>
      <td>0.445176</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.411344</td>
      <td>0.617469</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.434367</td>
      <td>0.674210</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-0.563221</td>
      <td>0.009331</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>190</th>
      <td>1.797756</td>
      <td>0.963394</td>
    </tr>
    <tr>
      <th>192</th>
      <td>-0.679177</td>
      <td>0.033222</td>
    </tr>
    <tr>
      <th>194</th>
      <td>0.975527</td>
      <td>0.041236</td>
    </tr>
    <tr>
      <th>196</th>
      <td>-1.354463</td>
      <td>0.450887</td>
    </tr>
    <tr>
      <th>198</th>
      <td>-2.341788</td>
      <td>0.009804</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>



위에 sampledf[::2]를 보시면 첫번째(index=0), 세번째(index=2), ...., 199번째(index=198)이 선택되는 것을 확인하실 수 있습니다. 2의 간격 크기로 행이 선택되는 것입니다.

<h1>df[::-1]</h1>

df[::-1] 인 경우는 열의 배치를 뒤집어줍니다. 아래는 예시입니다.


```python
sampledf[::-1]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>199</th>
      <td>2.600890</td>
      <td>0.775489</td>
    </tr>
    <tr>
      <th>198</th>
      <td>-2.341788</td>
      <td>0.009804</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-0.365103</td>
      <td>0.413758</td>
    </tr>
    <tr>
      <th>196</th>
      <td>-1.354463</td>
      <td>0.450887</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.685687</td>
      <td>0.933069</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.411344</td>
      <td>0.617469</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.703587</td>
      <td>0.718288</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.123720</td>
      <td>0.445176</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.208545</td>
      <td>0.459722</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.312234</td>
      <td>0.788584</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>



<h1>df[::-c]</h1>

마찬가지로, df[::-c] 이면 뒤에 row부터 2간격마다 row가 선택됩니다. 아래는 예시입니다.


```python
sampledf[::-2]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>199</th>
      <td>2.600890</td>
      <td>0.775489</td>
    </tr>
    <tr>
      <th>197</th>
      <td>-0.365103</td>
      <td>0.413758</td>
    </tr>
    <tr>
      <th>195</th>
      <td>0.685687</td>
      <td>0.933069</td>
    </tr>
    <tr>
      <th>193</th>
      <td>0.267967</td>
      <td>0.020342</td>
    </tr>
    <tr>
      <th>191</th>
      <td>-0.918194</td>
      <td>0.917082</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.924938</td>
      <td>0.837344</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.890616</td>
      <td>0.096270</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-0.603043</td>
      <td>0.697143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.703587</td>
      <td>0.718288</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.208545</td>
      <td>0.459722</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>


