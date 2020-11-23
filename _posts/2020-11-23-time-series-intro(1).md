---
layout : post
title: 1-1. Introduction to Time Series
category: Time Series Analysis
tags: time-series, time-series-analysis
---

이번 포스팅을 시작으로, 시계열 분석에 대해서 다루도록 하겠습니다. 메인 교재는 Brockwell와 Richard A. Davis의 \< Introduction to Time Series and Forecasting \> 와 패스트캠퍼스의 \<파이썬을 활용한 시계열 분석 A\-Z\> 를 듣고 정리하였습니다. 

<h2>1.1. Examples of Time Series</h2>

<h2>1.2. Objectives of Time Series Analysis</h2>

<h2>1.3. Some Simple Time Series Models</h2>

<h3>1.3.3 A General Approach to Time Series Modeling</h3>
시계열 분석에 대해 깊게 들어가기 전에, 시계열 데이터 모델링하는 방법에 대해 대략적으로 알아봅시다.

1) 그래프로 그린 후, 그래프 상에서 아래와 같은 요소가 있는지 체크한다.(Plot the series and examine the main features of the graph)<br> 
<ul><li>trend</li><li>a seasonal component</li><li>any apparent sharp changes in behavior</li><li>any outlying observations</li></ul>

2) 정상상태의 잔차를 얻기 위해, trend와 seasonality 요소를 제거한다. (Remove the trend and seasonal components to get stationary residuals)<br>
&nbsp;&nbsp;&nbsp;&nbsp; trend와 seasonality 요소를 제거하기 전에, 전처리를 해야하는 경우가 있습니다. 예를 들어, 아래와 같이 지수적으로 증가하는 경우에, 로그를 취해서 variance가 일정하도록 만든 후 모델링을 하면 정확도를 높일 수 있습니다.

<p align='center'><img src='https://imgur.com/V85l07h.png'><figcaption align='center'>그림 1. 로그 취하기 전</figcaption></p>
<p align='center'><img src='https://imgur.com/e0GKRKU.png'><figcaption align='center'>그림 1. 로그 취한 후</figcaption></p>

이외에도 여러 방법이 있습니다. 추후에 설명하도록 하겠습니다. 어쨌든, 이 모든 방법들의 핵심은 <b>정상상태의 잔차</b>를 만드는 것입니다.

3) auto-correlation 함수, 여러 다양한 통계량을 이용하여 잔차를 핏팅할 모델을 선택한다. (Choose a model to fit the residuals, making use of various sample statistics including the sample autocorrelation function)

4) 핏팅된 모델로 예측한다.<br> 
&nbsp;&nbsp;&nbsp;&nbsp; 여기서 잔차를 예측하는 것이고, 예측된 잔차를 원래 예측해야 할 값으로 변환한다.



