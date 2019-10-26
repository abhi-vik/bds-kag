# MIS382N Fall 2019 Kaggle competition

https://www.kaggle.com/c/mis382n-fall-2019

\- Abhilash Vikram Gupta

This will describe our day-to-day analyses and evolving strategies to get the highest possible ROC-AUC score on the data.

It is observed that this competition is clocked at UTC time. It makes sense to make 5 submissions each day ending 7 pm, and be sure to reflect upon what went right/wrong in some of the remaining time.

## Day 1

#### Ideas

* First glance: supervised binary classification with a heavy bias. Some features look like categorical variables, but our belief is that this is deceptive and that they are in fact count-like. Some of the other columns show heavy variation and others still are quite large (5-digit).
* Scaling followed by logistic regression gives a bad ROC-AUC score and is hopelessly biased towards the dominant label.
* A plain decision tree overfits hard. In fact, it gives an AUC of 1.0 on train data using just feature index 13 (f14).
* RFC does alright, but the score does not rise above 0.85. If we have aspirations of the highest possible score, we need to use gradient boosting.
* Considering LightGBM and XGBoost right now -- start with LightGBM because we have never used it and XGBoost is slow.
* Tuning LightGBM, key realizations:
  * LGB attempts to replicate scikit-learn's API.
  * We need to set class weights somewhere.
  * Use all available processors, always.
  * Focus heavily on controlling tree depth and learning depth (eta).

Day's best: 0.87978
Overall Best: 0.87978

#### Reflections

##### What went well
1. Got five submissions off despite short notice.
2. We seem to have the top spot, for now.

##### What we learnt
1. (Maybe) XGBoost appears to be more accurate than LightGBM but takes longer on larger datasets. One smart thing to do would be to use LightGBM on early on, when doing iterative testing, and use equivalent settings with XGBoost as the end of the competition draws near.
2. It is difficult to say which are the true features. It might be right to vary the features a little, taking 7 to 10 at a time and see how that fares.

##### How we can improve
1. Overrelying on LightGBM might leave us blind to other ways of solving the problem.
2. Need to learn more about decision trees and gradient boosting to tune better.
3. Could possibly use neural networks? Can investigate.


## Day 2

#### Ideas

* There seems to be a bit of an accuracy jump with features 0, 3, 7, 12, 13, 14, 15 and 16 that subsides when we add feature 6 to the mix.
* Spend the day attempting neural networks. Know that we cannot use the convolutional neural networks based predictions that we did in image classification. Have to try something else.
* Procured AUC scores of around 0.5 everytime. Spent some time fiddling, but then dropped it.

Day's best: ~0.45 (Obviously, something went very wrong. Still don't quite know what. Did not have time to investigate.)
Overall Best: 0.87978

#### Reflections

##### What went well
1. Very little. Using deep learning may be possible, but it will require time and patience.

##### What we learnt
1. Neural networks are terrible for classification with imbalanced datasets (at least, straight out of the box).
2. Focus on gradient boosting for now. Can come back to neural networks later.

##### How we can improve
1. Learning and using XGBoost is something that should definitely be done.
2. Spend more time with keras, etc. but later.


## Day 3

#### Ideas

* This is XGBoost day. Attempt to master the different parameters being used and beat the score set by LightGBM.
* It looks like XGBoost also supports the now familiar scikit-learn format api. Use that.

Day's best: 0.88192
Overall best: 0.88192

#### Reflections

##### What went well
1. A miniscule improvement in the best prediction set.
2. A better understanding of gradient boosting and XGBoost.

##### What we learnt
1. XGBoost is far, far slower than LightGBM. This means more time waiting for results and less time actually looking at them.
2. The increased AUC ROC score today may, in my opinion, be attributed to a combination of randomness and a better understanding of l1 and l2 regularization -- little else.
3. Unless there are other as of yet unknown benefits of XGBoost, we should prefer LightGBM in the long term.

##### How we can improve
1. Do some data engineering. Attempt polynomial features and interaction terms.


## Day 4

#### Ideas

* Spend the day on feature creation and decomposition.
* Attempt an auto-tuning process. Each loop should:
  * Find the most important features in the original dataset for the best current model.
  * Create new features that are exponential (eg. **0.5, **2) of these original features.
  * Scale the same original features and create interaction terms (added and subtracted).
  * Use recursive feature elimination with cross validation to select the best of all of the features combined.
  * Take the best parameters in the previously selected model. Create the parameters for a grid search by slightly altering the parameters.
  * Weigh the variation in parameters by multiplying the variation with a decomposition term -- a number that keeps decreasing until it hits zero.
  * Use the generated range of parameters to select the best model using a grid search with cross validation.
  * Decrement the decomposition term and repeat the process.

Day's best: null (Did not, in the end, submit any scores; had other things to attend to.)
Overall best: 0.88192

#### Reflections

##### What went well
1. Implemented the proposed auto-tuner. We now know that the design is possible and works, albeit inefficiently.
2. Experimented with some feature engineering.

##### What we learnt
1. The tool that we devised is inefficient, but automated. It could be used with success when given a dataset that we are not familiar with, or when one has other tasks. Focused human effort, however, yields better results.
2. Feature manipulation has almost no positive effect on our classifier. In fact, it even yields slightly lower scores.

##### How we can improve
1. Attempt to make five submissions a day (not submitting is quite bad).


## Day 5

#### Ideas
* We noticed earlier that for practical purposes, LightBGM proved to be as accurate and much faster than XGBoost. Spend some hours tuning it.

Day's best: 0.89364
Overall best: 0.89364

#### Reflections

##### What went well
1. Had the first significant improvement over our initial auc score. The overall improvement is just ~0.014, but that could prove significant.

##### What we learnt
1. Using the model's feature_importances_ attribute does not always yield the best features. This includes using feature selection methods that rely on the model's feature_importances_ attribute, such as recursive feature elimination.
2. We have yet to explore all the hyperparameters of gradient boosting. Learning more about them would be wise.

##### How we can improve
1. Study data, use intuition and handpick to obtain the best possible features.
2. Look further into the LightGBM documentation
