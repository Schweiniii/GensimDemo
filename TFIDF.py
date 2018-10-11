from gensim.models import word2vec
from gensim.models import tfidfmodel
from gensim.models import LsiModel
from gensim import corpora
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text8/text8')
dictionary = corpora.Dictionary(sentences)
dictionary.save_as_text('tfidf/dic')
bow_corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

# model = tfidfmodel.TfidfModel(bow_corpus)

# model.save('tfidf/tfidf.model')

lsi = LsiModel(bow_corpus, id2word=dictionary)
lsi.save('Lsi/lsi.model')