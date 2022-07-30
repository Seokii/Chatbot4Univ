import time
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import re
import gensim
from gensim.models import Word2Vec, KeyedVectors
from konlpy.tag import Komoran

df = pd.read_csv("../../변형데이터/통합본데이터.csv")
text = list(df['text'])

start = time.time()

# 형태소 분석기로 명사만 추출
print("학습할 텍스트 데이터에서 명사만 추출 시작")

komoran = Komoran()
docs = [komoran.nouns(sentence) for sentence in text]

print("명사 추출 완료 : ", time.time() - start)

print("fine-tuning 시작")

#fine-tuning gensim word2vec
model = gensim.models.Word2Vec.load("ko.bin")
model.wv.save_word2vec_format("ko.bin.gz", binary=False)

model_2 = Word2Vec(size=200, min_count=1)
model_2.build_vocab(docs)
total_examples = model_2.corpus_count
print(total_examples)

model_2.build_vocab([list(model.vocab.keys())], update=True)
model_2.intersect_word2vec_format("ko.bin.gz", binary=False)

model_2.train(docs, total_examples=total_examples, epochs=model_2.iter)
result = model_2.wv.most_similar('blm')
print(result)

model_2.save('ko_new.model')

print("fine-tuning 완료 : ", time.time()-start)
