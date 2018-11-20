# Agile User Story Point Estimation

## Table of contents
1. [Introduction](#introduction)
2. [Propose Solution](#propose-solution)
3. [Data](#data)
4. [Evaluation](#evaluation)

## Introduction

Effort estimation is an important part of software project management. Cost and schedule overruns create more risk for software projects. Effort estimations can benefit project manager and clients to make sure that projects can complete in time.

In modern Agile development, software is developed through repeated cycles (iterations). A project has a number of iterations. An iteration is usually a short (usually 2–4 weeks) period in which the development team designs, implements, tests and delivers a distinct product increment, e.g. a working milestone version or a working release. Each iteration requires the completion of a number of user stories, which are a common way for agile teams to express user requirements.

Therefore, there is a need to focus on estimating the effort of completing single user story rather than entire project. In fact, it is very common for agile teams to go through each user story estimate the effort required to completing it.

Currently, most agile teams heavily reply on experts' subjective assessment to estimate time. This may be bias and inaccuracy and inconsistent between estimates.

## Propose Solution

We would propose an end-to-end solution to estimate story point (number) based on user story (text).


- Input: User story (Title + description) of issues
- Output: story-point estimation

End-to-end approach, without worrying about manual features.

In order to do it, we would find a way to represent user story (Text to Vector), there are several ways to do it for example Doc2Vec model, LSTM, CNN or TF-IDF.

After that we can could use a regressor to estimate story point, some ideas could be Neural Network, Recurrent Highway Network, Gradient Boosting Tree.

## Data

Dataset are from 16 large open source projects in 9 repositories namely
- Apache
- Appcelerator
- DuraSpace
- Atlassian
- Moodle
- Lsstcorp
- Mulesoft
- Spring
- Tatendforge

The authors claim this is the largest dataset in term of number of data points for story point estimation

__Investigate data__

Dataset contain 23,313 issues with story points from 16 different projects.

Features:
- Story points
- title
- description


Among projects, 7 / 16 follow Fibonnaci scale, other 9 did not use any scale. Because of this reason, the authors did not round their estimation to nearest estimation on Fibonacci scale. This makes their approach applicable for wider range of projects.

## Evaluation

- Mean Absolute Error
- Median Absolute Error
- Standardized Accuracy

## Approach

<!-- Mean Absolute Error:  5.1271836124624
Median Absolute Error:  2.7
Mean Squared Error:  98.93675845009025 -->

Mean Absolute Error:  4.355169641701281
Median Absolute Error:  2.5006497967644896
Mean Squared Error:  80.21437110450948

Mean Absolute Error:  4.355169641701281
Median Absolute Error:  2.5006497967644896
Mean Squared Error:  80.21437110450948

Mean Absolute Error:  4.307588033064078
Median Absolute Error:  2.4698952283655884
Mean Squared Error:  78.65750135354794

Mean Absolute Error:  4.951236110767787
Median Absolute Error:  3.094385015535193
Mean Squared Error:  97.11308199781521

## Primitive Result

|Model|Mean Absolute Error|Median Absolute Error|Mean Square Error|
|:---:|:--:|:--:|:--:|
|Tf-idf + Random Forest|3.96|1.90|82.64|
|Tf-idf + LightGBM|4.41|2.43|84.59|
|tf-idf + Catboost -> cannot deal with sprase data type||||
|Average Word2Vec 100 + Random Forest|4.95|2.7|94.95|
|Average Word2Vec 100 + LightGBM|4.42|2.54|82.25|
|Average Word2Vec 200 + Random Forest|5.13|2.7|98.94|
|Average Word2Vec 200 + LightGBM|4.38|2.48|81.21|
|Average Word2Vec 300 + LightGBM|4.35|2.50|80.21|
|Average Word2Vec 400 + LightGBM|4.31|2.46|78.66|
|Average Word2Vec 400 + Catboost|4.73|3.16|90.31|
|Doc2Vec 100 + Random Forest|5.88|3.4|119.06|
|Doc2Vec 100 + LightGBM|4.84|3.05|97.44|
|LSTM end-to-end 50 unit tf.nn.rnn_cell.LSTMCell|3.97|N/A|92.07|
|LSTM end-to-end 50 unit, 100 embeddingsize tf.nn.rnn_cell.LSTMCell|3.98|N/A|90.51|
|LSTM end-to-end 100 unit tf.nn.rnn_cell.BasicLSTMCell|4.46|N/A|84.78|
|LSTM end-to-end 200 unit tf.nn.rnn_cell.LSTMCell|5.09|N/A|86.27|
|Average LSTM + Random Forest ||||
|Average LSTM + LightGBM ||||
|Average LSTM + Catboost ||||
|Average LSTM + Ensemble ||||
|Average LSTM + Recurrent Highway Network __(from paper)__ ||||
