# methods for data visualizations

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, ImageColorGenerator

# language counts data plot
def lang_plot(train_data):
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams.update({'font.size': 12})

    # graph of spanish vs. english data
    fig1 = sns.countplot(y='language', hue='language', legend=False, data = train_data, palette=['purple', 'green'])
    plt.title('Language Data Count')
    plt.show()

#graph of sexist vs. non-sexist count
def sexism_plot(train_data):
    fig1 = sns.countplot(x = 'task1', hue='task1', legend=False, data = train_data, palette=['red', 'blue'])
    plt.title('Sexist Data Count')
    plt.show()

#wordcount of sexist tweets
def count_sexist(train_data):
    sexist_tweets = train_data[train_data['task1'] == 'sexist'].copy()
    sexist_tweets['length'] = sexist_tweets['text'].apply(lambda x: len(x.split()))

    fig1 = sns.histplot(data=sexist_tweets, x='length', color='red')
    plt.title('Sexist Tweets Word Count Distribution')
    plt.xlabel('Word Count')
    plt.show()
    
#wordcount of nonsexist tweets
def count_nonsexist(train_data):
    nonsexist_tweets = train_data[train_data['task1'] == 'non-sexist'].copy()
    nonsexist_tweets['length'] = nonsexist_tweets['text'].apply(lambda x: len(x.split()))

    fig1 = sns.histplot(data=nonsexist_tweets, x='length', color='blue')
    plt.title('Non-Sexist Tweets Word Count Distribution')
    plt.xlabel('Word Count')
    plt.show()

# wordcloud of sexist tweets
def sexist_wordcloud(train_engtweets):
    sexist_words = ' '.join(text for text in train_engtweets['cleantext'][train_engtweets['task1']=='sexist'])

    wc = WordCloud(collocations = False, background_color='white', height=2000, width=2000).generate(sexist_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Sexist Tweets WordCloud')
    plt.show()

# wordcloud of nonsexist tweets
def nonsexist_wordcloud(train_engtweets):
    nonsexist_words = ' '.join(text for text in train_engtweets['cleantext'][train_engtweets['task1']=='non-sexist'])

    wc = WordCloud(collocations = False, background_color='white', height=2000, width=2000).generate(nonsexist_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Non-Sexist Tweets WordCloud')
    plt.show()