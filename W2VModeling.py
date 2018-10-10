from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text9/text9')

# 参数有min_count，制定词语的最小出现次数，默认是5；size是词向量的维度；
# 进行两次操作，第一次统计词频构建词典；第二次训练网络
model = word2vec.Word2Vec(sentences, size=200)

model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

model.save('text9/text.model')
# store the learned weights, in a format the original C tool understands
model.wv.save_word2vec_format('text9/text.model.bin',binary=True)