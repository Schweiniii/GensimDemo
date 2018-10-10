from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2VecKeyedVectors.load_word2vec_format('text8/text.model.bin', binary=True)

more_examples = ["he his she", "big bigger bad", "man men woman", "France Paris China", "America, American, China"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a,b,x,predicted))

Weirdo = model.wv.doesnt_match("breakfast cereal dinner lunch".split())
print(Weirdo)