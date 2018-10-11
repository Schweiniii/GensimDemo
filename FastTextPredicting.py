from gensim.models import FastText
from gensim.models import tfidfmodel
from gensim.models import lsimodel
from gensim import corpora

model = tfidfmodel.TfidfModel.load('tfidf/tfidf.model')
string = ['America', 'China']
dictionary = corpora.Dictionary.load_from_text('tfidf/dic')
string = dictionary.doc2bow(string)
print(model[string])

lsi = lsimodel.LsiModel.load('Lsi/lsi.model')
print(lsi[string])

'''more_examples = ["he his she", "big bigger bad", "man men woman", "France Paris China", "America, American, China"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print("'%s' is to '%s' as '%s' is to '%s'" % (a,b,x,predicted))'''