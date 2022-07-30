from flask import Flask, request, jsonify, abort, render_template
import socket
import json

# 챗봇 엔진 서버 정보
host = "127.0.0.1"      # 챗봇 엔진 서버 IP
port = 5050             # 챗봇 엔진 port

# Flask 애플리케이션
app = Flask(__name__)

# 챗봇 엔진 서버와 통신
def get_answer_from_engine(bottype, query):
    # 챗봇 엔진 서버 연결
    mySocket = socket.socket()
    mySocket.connect((host, port))

    # 챗봇 엔진 질의 요청
    json_data = {
        'Query' : query,
        'BotType' : bottype
    }
    message = json.dumps(json_data)
    mySocket.send(message.encode())

    # 챗봇 엔진 답변 출력
    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)

    # 챗봇 엔진 서버 연결 소켓 닫기
    mySocket.close()

    return ret_data

@app.route('/hello', methods=['GET'])
def index():
    message = "안녕하세요. 챗봇 질문데이터 수집기간 입니다.\n" \
              "현재 다음 기능을 제공하고 있습니다.\n" \
              "1.번호안내 2.장소안내\n" \
              "사용예시 - (학과/트랙/기관명)번호 알려줘, (건물이름)위치 알려줘\n" \
              "사용예시2 - 컴퓨터공학부 번호 알려줘, 상담실 위치 알려줘"

    json_data = {
        'message': message
    }
    message = json.dumps(json_data, ensure_ascii=False)
    message = json.loads(message)
    return jsonify(message)

    try:
        index()
    except Exception as ex:
        abort(500)

# 챗봇 엔진 query 전송 API
@app.route('/query/<bot_type>', methods=['POST'])
def query(bot_type):
    body = request.get_json()
    try:
        if bot_type == 'TEST':
            # 챗봇 API 테스트
            ret = get_answer_from_engine(bottype=bot_type, query=body['query'])
            return jsonify(ret)

        else:
            # 정의되지 않은 bot type인 경우 404 Error
            abort(404)

    except Exception as ex:
        # 오류 발생 시 500 Error
        abort(500)

# 이미지 테스트
@app.route('/images/<image_file>', methods=['GET'])
def image(image_file):
    return render_template('img.html', image_file='/resize_images/'+image_file+'_resize.jpg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)