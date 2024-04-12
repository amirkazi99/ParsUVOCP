import pandas as pd
from PersianStemmer import PersianStemmer
from hazm import POSTagger, WordEmbedding, word_tokenize
from hazm.utils import stopwords_list
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification


def getAdjectives(parsedComments):
    tagger = POSTagger(model='resources/pos_tagger.model', universal_tag=True)
    stemmer = PersianStemmer()
    adjectives = set()

    for parsedComment in parsedComments:
        for i in range(len(parsedComment)):
            token = parsedComment[i]
            adjectiveType = 0

            tag = token['tag']
            rel = token['rel']
            word = token['word']
            stemmed = stemmer.run(word) if word else None

            prevToken = parsedComment[i - 1] if i - 1 >= 0 else None
            nextToken = parsedComment[i + 1] if i + 1 < len(parsedComment) else None
            nextNextToken = parsedComment[i + 2] if i + 2 < len(parsedComment) else None
            nextNextNextToken = parsedComment[i + 3] if i + 3 < len(parsedComment) else None

            if tag in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'} and rel in {'PRD', 'NPOSTMOD', 'OBJ', 'ADV'}:
                adjectives.add(stemmed)
                adjectiveType = 1

            if tag in {'AJ', 'ADJ'} and rel in {'ROOT'}:
                adjectives.add(stemmed)
                adjectiveType = 2

            if prevToken and nextToken and tag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'}:
                prevTag = prevToken['tag']
                nextTag = nextToken['tag']

                if prevTag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'} and nextTag in {'V', 'VERB'}:
                    if tagger.tag([stemmed])[0][1] in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                        adjectives.add(stemmed)
                        adjectiveType = 3

            if nextToken and nextNextToken and nextNextNextToken and tag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'}:
                nextTag = nextToken['tag']
                nextNextTag = nextNextToken['tag']
                nextNextNextTag = nextNextNextToken['tag']

                if any(x in {'ADV'} for x in {nextTag, nextNextTag, nextNextNextTag}):
                    if all(x not in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'} for x in {nextTag, nextNextTag, nextNextNextTag}):
                        if tagger.tag([stemmed])[0][1] in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                            adjectives.add(stemmed)
                            adjectiveType = 4

            if adjectiveType != 0 and nextNextToken and nextToken['word'] in {'و'}:
                nextNextTag = nextNextToken['tag']
                nextNextWord = nextNextToken['word']
                nextNextStemmed = stemmer.run(nextNextWord) if nextNextWord else None

                if adjectiveType in {1} and nextNextTag in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                    adjectives.add(nextNextStemmed)

    stopwords = pd.read_excel('resources/stopwords.xlsx')['stopwords'].values.tolist() + stopwords_list()
    adjectives = [adjective for adjective in adjectives if adjective not in stopwords]
    return adjectives


def getNouns(parsedComments):
    tagger = POSTagger(model='resources/pos_tagger.model', universal_tag=True)
    stemmer = PersianStemmer()
    nouns = set()

    for parsedComment in parsedComments:
        hasObj = False
        for i in range(len(parsedComment)):
            if parsedComment[i]['rel'] in {'OBJ'}:
                hasObj = True
                break

        for i in range(len(parsedComment)):
            token = parsedComment[i]
            aspectType = 0

            tag = token['tag']
            rel = token['rel']
            word = token['word']
            stemmed = stemmer.run(word) if word else None

            nextToken = parsedComment[i + 1] if i + 2 < len(parsedComment) else None

            if hasObj:
                if rel in {'OBJ'} and tag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'}:
                    nouns.add(stemmed)
                    aspectType = 1
            else:
                if rel in {'SBJ'} and tag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'}:
                    nouns.add(stemmed)
                    aspectType = 2

            if rel in {'SBJ'} and tag in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                if tagger.tag([stemmed])[0][1] in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                    nouns.add(stemmed)
                    aspectType = 3

            if aspectType != 0 and nextToken and nextToken['word'] in {'و'}:
                nextNextToken = parsedComment[i + 2]

                nextNextTag = nextNextToken['tag']
                nextNextWord = nextNextToken['word']
                nextNextStemmed = stemmer.run(nextNextWord) if nextNextWord else None

                if aspectType in {1, 2} and nextNextTag in {'N', 'NOUN', 'Ne', 'NOUN,EZ'}:
                    nouns.add(nextNextStemmed)

                elif aspectType in {3} and nextNextTag in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                    if tagger.tag([stemmed])[0][1] in {'AJ', 'ADJ', 'AJe', 'ADJ,EZ'}:
                        nouns.add(nextNextStemmed)

    stopwords = pd.read_excel('resources/stopwords.xlsx')['stopwords'].values.tolist() + stopwords_list()
    nouns = [noun for noun in nouns if noun not in stopwords]
    return nouns


def removeNER(aspects):
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-ner-peyma")
    model = AutoModelForTokenClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-ner-peyma")
    classifier = pipeline("token-classification", model=model, tokenizer=tokenizer)

    results = classifier(aspects['aspect'].values.tolist())

    aspects = aspects[[len(result) == 0 for result in results]]
    return aspects


def createAspects(parsedComments, nouns, adjectives):
    stemmer = PersianStemmer()
    aspects = []

    for index, parsedComment in enumerate(parsedComments):
        nextWord = parsedComment[0]['word']
        nextStemmed = stemmer.run(nextWord) if nextWord else None
        nextNextWord = parsedComment[1]['word']
        nextNextStemmed = stemmer.run(nextNextWord) if nextNextWord else None

        multiWords = set()
        singleWords = []

        for i in range(len(parsedComment)):
            stemmed = nextStemmed

            if stemmed in nouns:
                # aspects.append((f'{stemmed}', index))
                singleWords.append(stemmed)

            if i < len(parsedComment) - 1:
                nextStemmed = nextNextStemmed

                if stemmed in nouns and nextStemmed in nouns:
                    aspects.append((f'{stemmed} {nextStemmed}', index))
                    multiWords.add(stemmed)
                    multiWords.add(nextStemmed)

                # if stemmed in nouns and nextStemmed in adjectives:
                #     aspects.append((f'{stemmed} {nextStemmed}', index))

                # if stemmed in adjectives and nextStemmed in nouns:
                #     aspects.append((f'{stemmed} {nextStemmed}', index))

            if i < len(parsedComment) - 2:
                nextNextWord = parsedComment[i + 2]['word']
                nextNextStemmed = stemmer.run(nextNextWord) if nextNextWord else None

                if stemmed in nouns and nextStemmed in nouns and nextNextStemmed in nouns:
                    aspects.append((f'{stemmed} {nextStemmed} {nextNextStemmed}', index))
                    multiWords.add(stemmed)
                    multiWords.add(nextStemmed)
                    multiWords.add(nextNextStemmed)

                if stemmed in nouns and nextStemmed in adjectives and nextNextStemmed in nouns:
                    aspects.append((f'{stemmed} {nextNextStemmed}', index))
                    multiWords.add(stemmed)
                    multiWords.add(nextNextStemmed)

        for word in singleWords:
            if word not in multiWords:
                aspects.append((f'{word}', index))

    aspects = pd.DataFrame(aspects, columns=['aspect', 'comment'])
    aspects = aspects.groupby('aspect')['comment'].apply(set).reset_index(name='comments')

    aspects = removeNER(aspects)

    aspects['frequency'] = aspects['comments'].apply(len)
    aspects.sort_values(by='frequency', ignore_index=True, inplace=True, ascending=False)
    aspects.to_excel('results/aspects.xlsx', index=False)
    aspects = aspects[aspects['frequency'] > aspects['frequency'].mean() + 0.5 * aspects['frequency'].std()]
    return aspects


def commentSentimentAnalyze(comments):
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
    model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased-sentiment-digikala")
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    results = classifier(comments)

    commentsSentiment = [(int(result['label'] == 'recommended') - int(result['label'] == 'not_recommended')) for result in results] # * result['score'] for result in results]
    return commentsSentiment


def matchCommentsAspects(comments, aspects):
    wordEmbedding = WordEmbedding(model_type='fasttext', model_path='resources/fasttext_skipgram_300.bin')
    stemmer = PersianStemmer()

    aspectsComments = aspects['comments'].values.tolist()
    aspects = aspects['aspect'].values.tolist()

    commentsAspectsMetrics = [[] for _ in range(len(comments))]

    for aspect, aspectComments in zip(aspects, aspectsComments):
        for aspectComment in aspectComments:
            if aspectComment < len(comments):
                commentsAspectsMetrics[aspectComment].append((aspect, 1.0))

    for index in range(len(comments)):
        tokens = word_tokenize(comments[index])
        aspectsMetrics = []
        for aspect in aspects:
            temp = []
            for token in tokens:
                temp.append(wordEmbedding.similarity(aspect, stemmer.run(token)))
            aspectsMetrics.append((aspect, 2. * max(temp) - 1.))

        aspectsMetrics.sort(key=lambda x: x[1], reverse=True)
        checklist = commentsAspectsMetrics[index].copy()
        for aspectMetric in aspectsMetrics:
            if aspectMetric[1] > 0.:
                if len(checklist) == 0 or all([aspectMetric[0] != item[0] for item in checklist]):
                    commentsAspectsMetrics[index].append(aspectMetric)
            else:
                break

    commentsAspects = pd.DataFrame({'comment': comments, 'aspects': [[] for _ in comments]})
    aspectsComments = pd.DataFrame({'aspect': aspects, 'comments': [[] for _ in aspects]})
    for index, commentAspectsMetrics in enumerate(commentsAspectsMetrics):
        for commentAspectMetric in commentAspectsMetrics:
            commentsAspects.at[index, 'aspects'].append(commentAspectMetric[0])
            aspectsComments.at[aspects.index(commentAspectMetric[0]), 'comments'].append(comments[index])

    return commentsAspectsMetrics, commentsAspects, aspectsComments


def aspectSentimentAnalyze(commentsSentiment, commentsAspects, aspects):

    aspects = aspects['aspect'].values.tolist()
    sentiments = [0 for _ in aspects]
    weights = [0 for _ in aspects]

    for commentSentiment, commentAspects in zip(commentsSentiment, commentsAspects):
        for aspect, metric in commentAspects:
            index = aspects.index(aspect)
            sentiments[index] += metric * commentSentiment
            weights[index] += 1

    sentiments = [sentiments[i] / weights[i] if weights[i] > 0 else None for i in range(len(sentiments))]

    aspectsSentiment = pd.DataFrame({'aspect': aspects, 'sentiments': sentiments})
    aspectsSentiment.dropna(inplace=True)
    aspectsSentiment.sort_values('sentiments', ascending=False, inplace=True, ignore_index=True)

    return aspectsSentiment
