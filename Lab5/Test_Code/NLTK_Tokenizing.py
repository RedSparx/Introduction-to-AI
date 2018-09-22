from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents


tweets = twitter_samples.strings('positive_tweets.json')
tweets_tokens = twitter_samples.tokenized('positive_tweets.json')

tweets_tagged = pos_tag_sents(tweets_tokens)
print(tweets_tagged)

adjectives = 0
nouns = 0

for tweet in tweets_tagged:
    print(tweet)
    for data in tweet:
        tag = data[1]
        if tag == 'JJ':
            adjectives+=1
        if tag == 'NN':
            nouns+=1

print('Adjectives: %d'%adjectives)
print('Nouns: %d'%nouns)
