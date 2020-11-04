---
title: "Movie Recommendation System"
date: "2020-11-04"
tages: [algorithm, recommender system , movie]
excerpt: "Algorithm, Recommender System"
categories:
- Algorithm
- Recommender System
- Analytics
---

# About
In this kernel we'll be building a baseline Movie Recommendation System using TMDB 5000 Movie Dataset.

There are basically three types of recommender systems:-

**Demographic Filtering**- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.

**Content Based Filtering**- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.

**Collaborative Filtering**- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.

I will be coding the first two of them since our dataset does not have userid.

Lets import the data.
```python
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)

df1 = pd.read_csv("tmdb_5000_credits.csv")
df2 = pd.read_csv("tmdb_5000_movies.csv")
```
We can merge them on id column.
```python
df1.columns = ['id', 'title', 'cast', 'crew']
df2 = df2.merge(df1, on='id')
```

## Demographic Filtering
```python
# We need a metric to rank the movies.
# vote_average itself is not enough to score fairly. Since we cannot rely on vote averages based on 2-3 votes. So
# We need a more robust metric. We can use IMDB's metric for that. It is as follows:
# Weighted Rating (WR) = (v/(v+m) * R)+(m/(v+m) * C)
# where,
# v is the number of votes for the movie;
# m is the minimum votes required to be listed in the chart;
# R is the average rating of the movie; And
# C is the mean vote across the whole report
```

```python
#We have v and v as vote_count and R as vote_average. we can calculate C as
C = df2["vote_average"].mean()
df2["vote_count"].describe()
m = df2["vote_count"].quantile(0.9)
```

```python
#Qualified movies list according to minimum votes required to be listed
q_movies = df2[df2["vote_count"]>m]
```
Now we have to define a scoring metric. Since some movies are voted from 2-3 people, it not fair to accept their rating as it is.
The following is the default weighted average scoring metric ıf IMDB.
```python
def score_movie(movie, m=m, C=C):
    R = movie["vote_average"]
    v = movie["vote_count"]
    return (v/(v+m))*R + (m/(v+m)*C)
```

Now we can add the scores to the qualified movies table
```python
q_movies["score"] = q_movies.apply(score_movie, axis=1)
print(q_movies["score"])
```
Sorting the movies based on weighted score
```python
q_movies = q_movies.sort_values(by="score", ascending=False)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/recommend/recommend_1.png" alt="sorted demographic recommendations">

## Content Based Filtering
For this, we will find similar movies according to their overview, cast, crew, keyword, tagline etc.

```python
"""
For any of you who has done even a bit of text processing before knows we need to convert the word vector of each overview.
Now we'll compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each overview.

Now if you are wondering what is term frequency , it is the relative frequency of a word in
a document and is given as (term instances/total instances). Inverse Document Frequency is the relative count
of documents containing the term is given as log(number of documents/documents with term) The overall importance
of each word to the documents in which they appear is equal to TF * IDF

This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear
in at least one document) and each row represents a movie, as before.This is done to reduce the importance of words
that occur frequently in plot overviews and therefore, their significance in computing the final similarity score.

Fortunately, scikit-learn gives you a built-in TfIdfVectorizer class that produces the TF-IDF matrix in a couple of lines.
That's great, isn't it?
"""
```
Computing tfidf matrix with sklearn
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words ='english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
print(tfidf_matrix.shape)
```
Find similarities between word matrix with cosine similarity
```python
from sklearn.metrics.pairwise import linear_kernel
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```

Construct a reverse map of indices and movie titles
```python
indices = pd.Series(df2.index, index=df2['title_x']).drop_duplicates()
```
Function that takes in movie title as input and outputs most similar movies
```python
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df2['title_x'].iloc[movie_indices]
```

Lets see the recommendations
```python
print(get_recommendations('The Dark Knight Rises'))
print(get_recommendations('The Avengers'))

```

<img src="{{ site.url }}{{ site.baseurl }}/images/recommend/recommend_2.png" alt="content based recommendations">

These doesn't look so good. We can improve it by including credits, genre and keywords to our recommender.

Parse the stringified features into their corresponding python objects
```python
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
```

We can get director of a movie given a row.
```python
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
        return np.nan
```

We can get top 3 elements of a list, if the entry has more than 3 items.
```python
# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
```

Define new director, cast, genres and keywords features that are in a suitable form.
```python
df2['director'] = df2['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
```

Print the new features of the first 3 films
```python
df2[['title_x', 'cast', 'director', 'keywords', 'genres']].head(3)
```

```python
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```

Apply clean_data function to your features.
```python
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
```

Now we will concat our columns to prepare it for the Count Vectorizer.
```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1)
```
Import CountVectorizer and create the count matrix
```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
```
Compute the Cosine Similarity matrix based on the count_matrix
```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
```
Reset index of our main DataFrame and construct reverse mapping as before
```python
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title_x'])


print(get_recommendations('The Avengers', cosine_sim2))
print(get_recommendations('The Avengers', cosine_sim))
```

<img src="{{ site.url }}{{ site.baseurl }}/images/recommend/recommend_3.png" alt="more based based recommendations">
