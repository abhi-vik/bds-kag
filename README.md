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

#### How we can improve:

1. Overrelying on LightGBM might leave us blind to other ways of solving the problem.
2. Need to learn more about decision trees and gradient boosting to tune better.
3. Could possibly use neural networks? Can investigate.

## Day 2:

* There seems to be a bit of an accuracy jump with features 0, 3, 7, 12, 13, 14, 15 and 16 that subsides when we add feature 6 to the mix.
