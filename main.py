import json
import time

import requests
from hazm import Normalizer, Lemmatizer, DependencyParser, sent_tokenize, word_tokenize

from Digikala import getComments, getCategoryProducts
from RuleFunctions import *


def getToken():
    apikey = "bd8a6b8f-c440-ee11-9abe-d397223d3fbe"

    baseUrl = "http://api.text-mining.ir/api/"
    url = baseUrl + "Token/GetToken"
    querystring = {"apikey": apikey}  # replace YOUR_API_KEY
    response = requests.request("GET", url, params=querystring)
    data = json.loads(response.text)
    return data['token']


def informalToFormal(data, tokenKey):

    baseUrl = "http://api.text-mining.ir/api/"
    url = baseUrl + "TextRefinement/FormalConverter"
    headers = {
        'Authorization': "Bearer " + tokenKey,
        'Cache-Control': "no-cache"
    }

    response = requests.request("POST", url, json=data, headers=headers)
    result = response.text
    response.close()
    return result


def getDigikalaComments(productCode):

    categoryCodes = getCategoryProducts(productCode, 200)

    productComments = getComments([productCode], 27)
    categoryComments = getComments(categoryCodes, 200, 10)

    seperator = len(productComments)
    comments = productComments + categoryComments

    return comments, seperator


def preprocess(comments, seperator):
    # normalize
    normalizer = Normalizer()
    preprocessedComments = []
    for comment in comments[:seperator]:
        for sentence in sent_tokenize(comment.replace('\r', '')):
            preprocessedComments.append(normalizer.normalize(sentence))

    newSeperator = len(preprocessedComments)
    for comment in comments[seperator:]:
        for sentence in sent_tokenize(comment.replace('\r', '')):
            preprocessedComments.append(normalizer.normalize(sentence))

    # pinglish
    # TODO

    # informal
    tokenKey = getToken()
    preprocessedComments2 = []
    for preprocessedComment in preprocessedComments:
        preprocessedComments2.append(informalToFormal(preprocessedComment, tokenKey))
    preprocessedComments = preprocessedComments2

    # spell
    # TODO

    # dependency parser
    tagger = POSTagger('resources/pos_tagger.model', universal_tag=True)
    lemmatizer = Lemmatizer()
    parser = DependencyParser(tagger=tagger, lemmatizer=lemmatizer)

    parsedComments = []
    for comment in preprocessedComments:
        parsedComments.append(parser.parse(word_tokenize(comment)).nodes)

    return preprocessedComments[:newSeperator], parsedComments


if __name__ == '__main__':
    start_time = time.time()

    comments, seperator = getDigikalaComments(productCode= 6352136)

    comments, parsedComments = preprocess(comments, seperator)

    adjectives = getAdjectives(parsedComments)

    nouns = getNouns(parsedComments)

    aspects = createAspects(parsedComments, nouns, adjectives)

    commentsSentiment = commentSentimentAnalyze(comments)

    commentsAspectsMetrics, commentsAspects, aspectsComments = matchCommentsAspects(comments, aspects)

    aspectsSentiment = aspectSentimentAnalyze(commentsSentiment, commentsAspectsMetrics, aspects)

    commentsAspects.to_excel('results/comments_aspects.xlsx', index=False)
    aspectsComments.to_excel('results/aspects_comments.xlsx', index=False)
    aspectsSentiment.to_excel('results/aspects_sentiment.xlsx', index=False)

    end_time = time.time()
    print(f'--- {end_time - start_time:.2f} seconds ---')