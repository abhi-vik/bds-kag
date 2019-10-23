# MIS382N Fall 2019 Kaggle competition

https://www.kaggle.com/c/mis382n-fall-2019

This will describe my day-to-day analyses and evolving strategies to get the highest possible ROC-AUC score.

It is observed that this competition is clocked at UTC time. It makes sense to make 5 submissions each day ending 7 pm, and be sure to reflect upon what went right/wrong in some of the remaining time.

## Day 1:

### Day's best: 0.87978

### Overall Best: 0.87978

### Reflections:

#### Lessons learnt:

1. (Maybe) XGBoost appears to be more accurate than LightGBM but takes longer on larger datasets. One smart thing to do would be to use LightGBM on early on, when doing iterative testing, and use equivalent settings with XGBoost as the end of the competition draws near.

2. It is difficult to say which are the true features. It might be right to vary the features a little, taking 7 to 10 at a time and see how that fares.

#### What went well:

1. Got five submissions off despite short notice.

2. We seem to have the top spot, for now.
