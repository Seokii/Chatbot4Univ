from PIL import Image
from io import BytesIO
import glob
import os

# 현재 경로와 저장 경로 설정
path = os.path.dirname(os.path.realpath(__file__))
save_path = "/home/irlab/hoseochatbot/chatbot_api/static/resize_images/"
os.chdir(path)
print("path is : " + path)

# 세이브 경로의 모든 jpg 파일 삭제
[os.remove(f) for f in glob.glob(save_path+"*.jpg")]

# 배열 선언
image_list_png = []
image_list_jpg = []

# png 파일 불러오기 + jpg 변환
read_files_png = glob.glob("/home/irlab/hoseochatbot/chatbot_api/static/images/*.png")
read_files_png.sort()
print(read_files_png)
print(len(read_files_png))

# .png 뺀 파일명 추출
image_list = os.listdir("/home/irlab/hoseochatbot/chatbot_api/static/images")
image_list.sort()
print(image_list)
print(len(image_list))
search = '.png'
for i, word in enumerate(image_list):
    if search in word:
        image_list_png.append(word.strip(search))
search = '.jpg'
print(image_list_png)
print(len(image_list_png))
print(image_list)
print(len(image_list))

# png to jpg
cnt2 = 0
for f in read_files_png:
    img = Image.open(f).convert('RGB')
    img.save("/home/irlab/hoseochatbot/chatbot_api/static/images/"+image_list_png[cnt2]+".jpg", 'jpeg')
    cnt2 += 1

# jpg 파일 resizing
read_files_jpg = glob.glob("/home/irlab/hoseochatbot/chatbot_api/static/images/*.jpg")
read_files_jpg.sort()

# .jpg 뺀 파일명 추출
image_list = os.listdir("/home/irlab/hoseochatbot/chatbot_api/static/images")
image_list.sort()
print(image_list)
print(len(image_list))
for i, word in enumerate(image_list):
    if search in word:
        image_list_jpg.append(word.strip(search))
print(image_list_jpg)
print(len(image_list_jpg))

cnt=0
for f in read_files_jpg:
    print(f)
    img = Image.open(f)
    buffer = BytesIO()
    img.save(buffer, 'jpeg', quality=70)
    buffer.seek(0)
    with open(save_path + image_list_jpg[cnt] + '_resize.jpg', 'wb') as nfile:
        nfile.write(buffer.getvalue())
    cnt += 1
