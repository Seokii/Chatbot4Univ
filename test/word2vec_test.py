from gensim.models import Word2Vec

# load model
model = Word2Vec.load("../train_tools/qna/ko.bin")
print(f"corpus total words : {model.corpus_total_words}")

# test model
print(f"사랑 : {model.wv['사랑']}")

# test similarity
print(f"일요일 = 월요일\t {model.wv.similarity(w1='일요일', w2='월요일')}")
print(f"엄마 = 아빠\t {model.wv.similarity(w1='엄마', w2='아빠')}")
print(f"엄마 = 어머니\t {model.wv.similarity(w1='엄마', w2='어머니')}")

# most similar words
print(model.wv.most_similar("엄마", topn=5))
print(model.wv.most_similar("삼성", topn=5))