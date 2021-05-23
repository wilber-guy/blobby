# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:18:33 2021

@author: 17013
"""


import nltk
from nltk.corpus import wordnet as wn
import wikipedia as wiki
import spacy
import yake
import gensim

# https://towardsdatascience.com/visualizing-networks-in-python-d70f4cbeb259
# https://hub.gke2.mybinder.org/user/westhealth-pyvis-fc5gtj65/notebooks/notebooks/example.ipynb#
from pyvis.network import Network
import networkx as nx


g = Network(height='800px',width='1200px',directed=False)
g.add_nodes(range(5))
g.add_edges([
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 1),
    (1, 3),
    (1, 2)
])
g.barnes_hut()
g.show("example.html")

# Hyper Parameters for packages
nlp = spacy.load("en_core_web_lg")

language = "en"
kw_extractor = yake.KeywordExtractor()
max_ngram_size = 4
deduplication_threshold = 0.3
numOfKeywords = 30
custom_kw_extractor = yake.KeywordExtractor(lan=language,
                                            n=max_ngram_size,
                                            dedupLim=deduplication_threshold,
                                            top=numOfKeywords,
                                            features=None)
'''


synonyms = []
antonyms = []

def get_syn_ant(word):
    word_syns = wn.synsets(word)
    if len(word_syns) > 0:
        for syn in word_syns:
            for l in syn.lemmas():
                synonyms.append(l.name())
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())

        print(word)
        print('--------------')
        print('Synonyms:', synonyms)
        print('Antonyms:', antonyms)
        print('Definition:', word_syns[0].definition())
        print('Examples:', word_syns[0].examples())

        print('Similarity of {} and {}: '.format(word_syns[0].lemmas()[0].name(), word_syns[1].lemmas()[0].name()), word_syns[0].wup_similarity(word_syns[1]) )

    else:
        print('nothing found for {}'.format(word))

get_syn_ant('ship')


w1 = wn.synset('plane.n.01')
w2 = wn.synset('boat.n.01')
print(w1.wup_similarity(w2))

'''

def extract_named_entities(text):
    # https://spacy.io/usage/linguistic-features#named-entities
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        ents.append(set(print(ent.text, ent.start_char, ent.end_char, ent.label_)))

    return ents

def get_wiki(topic):
    # https://pypi.org/project/wikipedia/
    text = wiki.summary(topic)
    search =wiki.search(topic)

    return (text, search)


def yake_keywords(text):
    #https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c
    keywords = custom_kw_extractor.extract_keywords(text)

    return keywords

def main():
    graph = Network(height='800px',width='1200px',directed=False)

    text, search = get_wiki('Ronald McDonald')
    yake_kw = yake_keywords(text)
    ents = extract_named_entities(text)

    print(yake_kw, '\n')
    print(ents, '\n')
    print(search, '\n')
    print(text, '\n')



if '__main__' == main():
    main()

'''
def most_similar(word, topn=5):
    word = nlp.vocab[str(word)]
    queries = [
        w for w in word.vocab
        if w.is_lower == word.is_lower and w.prob >= -15 and nlp.count_nonzero(w.vector)
    ]

    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return [(w.lower_,w.similarity(word)) for w in by_similarity[:topn+1] if w.lower_ != word.lower_]

most_similar("dog", topn=3)



nlp = spacy.load("en_core_web_md")



 # make sure to use larger package!
doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

# Similarity of two documents
print(doc1, "<->", doc2, doc1.similarity(doc2))
# Similarity of tokens and spans
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))

https://spacy.io/usage/spacy-101#vectors-similarity
def two_sentences(sent1, sent2):
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)


    sent_sim = doc1.similarity(doc2)

    print(sent_sim)



'''