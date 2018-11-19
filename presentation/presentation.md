
# Project Proposal

## Table of contents
1. [Introduction](#introduction)


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

![](pic/Table1.png)

## How they build dataset:

__Collect data__

JIRA Agile plugin is an issue tracking system that supports Agile development.

The authors use REST API from JIRA to query and collect those issue reports up until August 8, 2016.



__Traing / Test split__

To mimic a real deployment scenario that prediction on a current issue is made by using knowledge from estimations of the past issues, the issues in each project were split into training set (60% of the issues), development/validation set (i.e. 20%), and test set (i.e. 20%) based on their creation time. The issues in the training set and the validation set were created before the issues in the test set, and the issues in the training set were also created before the issues in the validation set.

Is dataset sorted by creation time?

## Approach

- Input: Title + description of an issues
- Output: story-point estimation

End-to-end approach, without worrying about manual features.

LSTM + Recurrent Highway Network (RHN)

![](pic/architecture.png)

- LSTM: play a role of represent documents, aka DOC2VEC -> How about DOC2VEC model (Google paper).
- Recurrent Highway Net:

## Evaluation

- Mean Absolute Error
- Median Absolute Error
- Standardized Accuracy

To compare the performance of two models, the authors tested statisical significance of 2 models using Wilcoxon Signed Ranked Test (it is a safe test because it makes no assumption about data distribution).

Null hypothesis: "the absolute errors provided by an estimation model are not different to those provided by another estimation model"
