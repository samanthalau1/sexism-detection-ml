# Twitter Sexism Detection Machine Learning Project

## Overview
As the world  sees a rapid proliferation in the use of social media, there is an increased risk of exposure to harmful comments, including sexist ones. Sexism is a hateful concept that can continue to corrupt minds if left unchecked. This project aims to identify if a machine learning model can proficiently identify sexism in tweets.

## Research Poster
<img src="Sexism Research.png" alt="Research Poster" width="700"/>

## Model and Methodology
### Approach
- Preprocess dataset of tweets
- Visualization of the data
- Utilize featurization techniques including TF-IDF and Word2Vec
- Tune the hyperparameters of Word2Vec
- Train a Logistic Regression model on the featurization methods to evaluate their performance
- Tune the hyperparameters of Logistic Regression model
- Compare to Random Forest Classifier model
- Evaluation and visualization of performance

## Results
Using Logistic Regression, TF-IDF demonstrated slightly better accuracy and F1 scores compared to Word2Vec. TF IDF achieved an accuracy of 71% compared to Word2Vec's 66% and an F1 score of 71% compared to Word2Vec's 64%.

## References
Das, Deepak. “Social Media Sentiment Analysis using Machine Learning : Part — I.” Towards Data Science, 6 September 2019
Fioto, Alex. “Classification Visualizations with Yellowbrick” Medium, 4 October 2020.  
Kant, Rajni. “Twitter Hate Speech Analysis.” Kaggle, 2021.  
Shaikh, Javed. “Machine Learning, NLP: Text Classification using scikit-learn, python and NLTK.” Towards Data Science, 23 July 2017.  
Van Otten, Neri. “Word2Vec For Text Classification [How To In Python & CNN].” Spot Intelligence, 15 February 2023.  