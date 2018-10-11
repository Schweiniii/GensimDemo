from gensim.models import word2vec
from gensim.models import fasttext
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text8/text8')

model = fasttext.FastText(sentences)

model.save('FT8/fasttext.model')