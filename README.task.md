# Practical Machine Learning and Deep Learning - Assignment 2 - Movie Recommender System

## Task description

A recommender system is a type of information filtering system that suggests items or content to users based on their interests, preferences, or past behavior. These systems are commonly used in various domains, such as e-commerce, entertainment, social media, and online content platforms.

Your assignment is to create a recommender system of movies for users:
* Your system should suggest some movies to the user based on user's gemographic information(age, gender, occupation, zip code) and favorite movies (list of movie ids).
* Solve this task using a machine learning model. You may consider only one model: it will be enough.
* Create a benchmark that would evaluate the quality of recommendations of your model. Look for commonly used metrics to evaluate a recommender system and use at least one metric.
* Make a single report decribing data exploration, solution implementation, training process, and evaluation on the benchmark.
* Explicitly state the benchmark scores of your systems.

Submission should be a link to GitHub repository. It should be open repository, so that the instructors could assess it easily.

## Data Description

In this assignment you will use [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) consisting user ratings to movies.

**General information about the dataset:**
* It consists of 100,000 ratings from 943 users on 1682 movies
* Ratings are ranged from 1 to 5
* Each user has rated at least 20 movies
* It contains simple demographic info for the users (age, gender, occupation, zip code)

**Detailed description of data files:**

| **File** | **Description** |
| -------- | --------------- |
| u.data | Full dataset of 100000 ratings by 943 users on 1682 items. Users and items are numbered consecutively from 1. The data is randomly ordered. This is a tab separated list of user id, item id, rating, and timestamp. The time stamps are unix seconds. |
| u.info | The number of users, items, and ratings in the u data set |
| u.item | Information about the items (movies). This is a tab separated list of movie id, movie title, release date, video release date, IMDB URL, and genres. The last 19 fields are genres and contain binary values. Movies can be of several genres at once. The movie ids are the ones used in u.data |
| u.genre | List of genres. |
| u.user | Demographic information about the users. This is a tab separated list of user id, age, gender, occupation, zip code. The user ids are the ones used in in u.data file. |
| u.occupation | List of occupations. |
| u1.base, u1.test, u2.base, u2.test, u3.base, u3.test, u4.base, u3.test, u5.base, u5.test | The data sets u1.base and u1.test through u5.base and u5.test are 80%/20% splits of the u data into training and test data. Each of u1, ..., u5 have disjoint test sets; this if for 5 fold cross validation (where you repeat your experiment with each training and test set and average the results). These data sets can be generated from u.data by mku.sh. |
| ua.base, ua.test, ub.base, ub.test | The data sets ua.base, ua.test, ub.base, and ub.test split the u data into a training set and a test set with exactly 10 ratings per user in the test set. The sets ua.test and ub.test are disjoint. These data sets can be generated from u.data by mku.sh. |
| allbut.pl | The script that generates training and test sets where all but n of a users ratings are in the training data |
| mku.sh | A shell script to generate all the u data sets from u.data. |

## Evaluation criterias

The repository should have the following structure:

```
movie-recommender-system
├── README.md               # The top-level README
│
├── data
│   ├── external            # Data from third party sources
│   ├── interim             # Intermediate data that has been transformed.
│   └── raw                 # The original, immutable data
│
├── models                  # Trained and serialized models, final checkpoints
│
├── notebooks               #  Jupyter notebooks. Naming convention is a number (for ordering),
│                               and a short delimited description, e.g.
│                               "1.0-initial-data-exporation.ipynb"            
│ 
├── references              # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports
│   ├── figures             # Generated graphics and figures to be used in reporting
│   └── final_report.pdf    # Report containing data exploration, solution exploration, training process, and evaluation
│
└── benchmark
    ├── data                # dataset used for evaluation 
    └── evaluate.py         # script that performs evaluation of the given model
```


In the top `README.md` file put your name, email and group number.

In the `reports` directory create a report about your work. In the report, describe in details the implementation of your system. Mention its advantages and disadvantages.

### Expected Report Structure

```
# Introduction
...
# Data analysis
...
# Model Implementation
...
# Model Advantages and Disadvantages
...
# Training Process
...
# Evaluation
...
# Results
...
```

In the `notebooks` directory put at least two notebooks. **First notebook** should contain your initial data exploration and basic ideas behind data preprocessing. **Second notebook** should contain information about final solution training and visualization.

## Grading criterias

Full assignment without any problems is said to be the `100%` solution.

| Criteria | Weight (%) | Comment |
| ---- | ----- | ----- |
| Structure and code quality | 30 | Code quality, structure, comments, clean repo, commit history, reproducibility (manual seeding) |
| Visualization, notebooks quality | 10 | Jupyter notebooks, visualizations |
| Solution building | 40 |  Implementation description, references, final report structure |
| Final score, evaluation  | 20 | Evaluation function, final score, quality of results |

If **PMLDL Course Team** will have any questions about your assignment or your work fails to show your results you will be called solution defence procedure. 


