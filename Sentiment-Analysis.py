print("Sentiment Analysis")

import pandas as pd
data = pd.read_csv('https://docs.google.com/spreadsheets/d/1F15gF01liemxjR101uY9h52J1rUkwawwS70cmTMWYtU/pub?gid=400689247&single=true&output=csv')
tweet_data = data["text"]
tweet_text = " ".join([text for text in tweet_data])


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk.tokenize.casual import (TweetTokenizer, casual_tokenize)
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
from nltk import Text

text_tokens = casual_tokenize(tweet_text)
lower_tokens = [word.lower() for word in text_tokens if len(word) >1]
word_tokens = [word for word in lower_tokens if (word.isalpha() and not word.startswith("http")) and not word.startswith("@") and not word.startswith("#")]
stopps = stopwords.words("english") + ["rt", "via", "ping", "@:", "15", "10"]
unstop_tokens = [word for word in word_tokens if word not in stopps]
nltk_tokens = Text(unstop_tokens)

hashtags = [word for word in lower_tokens if word.startswith("#")]
ats = [word for word in lower_tokens if word.startswith("@")]

token_counts = Counter(unstop_tokens)
most_common_tokens = [token for token, count in token_counts.most_common(30)]
hashtag_counts = Counter(hashtags)
at_counts = Counter(ats)

freq_dist = FreqDist(unstop_tokens)
freq_dist.plot(30, cumulative=False, title="Top 'welfare' word frequencies")
plt.savefig('frequency_disttribution.png')

fig = nltk_tokens.dispersion_plot(most_common_tokens)
plt.xticks([])
plt.savefig('dispersion_plot.png', labels=[])

print("\n :: #Tags ::")
for tag in hashtag_counts.most_common(10):
    print(tag[0][1:])

print("\n :: Users ::")
for user in at_counts.most_common(10):
    print(user[0][1:])

print("\n :: Collocations ::")
nltk_tokens.collocations(window_size=10, num=20)
