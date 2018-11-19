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

## Primitive Result

|Model|Mean Absolute Error|Median Absolute Error|Mean Square Error|
|:---:|:--:|:--:|:--:|
|Tf-idf + Random Forest|3.96|1.90|82.64|
|Tf-idf + LightGBM|4.41|2.43|84.59|
|tf-idf + Catboost -> cannot deal with sprase data type||||
|Average Word2Vec + Random Forest||||
|Average Word2Vec + LightGBM||||
|Average Word2Vec + Catboost||||
|LSTM end-to-end||||
|Average LSTM + Random Forest ||||
|Average LSTM + LightGBM ||||
|Average LSTM + Catboost ||||
|Average LSTM + Ensemble ||||
|Average LSTM + Recurrent Highway Network __(from paper)__ ||||
