---
layout : post
title: A Dual-Stage Attention-Based Recurrent Neural Network for Time-Series Prediction 논문 리뷰
category: Time Series Analysis
tags: time-series time-series-analysis prediction stock-prediction 
---

A Dual-Stage Attention-Based Recurrent Neural Network는 다변량 시계열 예측 모델입니다(Multi-Variate Time Series Prediction). Bahdanau et al.의 [Attention network 기반 시퀀스 모델](https://arxiv.org/abs/1409.0473) 을 베이스로, 인코더 뿐만 아니라 디코더에도 Attention netowork를 이용해 예측을 위한 다변량 시계열 변수 간 상대적인 중요도와 타임 스텝 간 상대적인 중요도를 모두 고려한 모델입니다. 

<h2>Problem</h2>

