import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
embedding_data = torch.load('C:/Users/Home/Documents/GitHub/Chatbot4Univ/train_tools/qna/embedding_data.pt')
df = pd.read_excel('C:/Users/Home/Documents/GitHub/Chatbot4Univ/train_tools/qna/train_data.xlsx')

# 질문 예시 문장
sentence = "컴공 과사 번호 알려줘"
print("질문 문장 : ",sentence)
sentence = sentence.replace(" ","")
print("공백 제거 문장 : ", sentence)

# 질문 예시 문장 인코딩 후 텐서화
sentence_encode = model.encode(sentence)
sentence_tensor = torch.tensor(sentence_encode)

# 저장한 임베딩 데이터와의 코사인 유사도 측정
cos_sim = util.cos_sim(sentence_tensor, embedding_data)
print(f"가장 높은 코사인 유사도 idx : {int(np.argmax(cos_sim))}")

# 선택된 질문 출력
best_sim_idx = int(np.argmax(cos_sim))
selected_qes = df['질문(Query)'][best_sim_idx]
print(f"선택된 질문 = {selected_qes}")

# 선택된 질문 문장에 대한 인코딩
selected_qes_encode = model.encode(selected_qes)

# 유사도 점수 측정
score = np.dot(sentence_tensor, selected_qes_encode) / (np.linalg.norm(sentence_tensor) * np.linalg.norm(selected_qes_encode))
print(f"선택된 질문과의 유사도 = {score}")

# 답변
answer = df['답변(Answer)'][best_sim_idx]
imageUrl = df['답변 이미지'][best_sim_idx]
print(f"\n답변 : {answer}\n")
print(f"답변 이미지 : {imageUrl}")