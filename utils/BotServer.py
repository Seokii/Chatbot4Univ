import socket

class BotServer:
    def __init__(self, srv_port, listen_num):
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None

    # sock 생성
    def create_sock(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 소켓 닫아도 바로 사용가능하게 설정
        self.mySock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.mySock.bind(("0.0.0.0", int(self.port)))
        self.mySock.listen(int(self.listen))
        return self.mySock

    # client 대기
    def ready_for_client(self):
        return self.mySock.accept()

    # sock 반환
    def get_sock(self):
        return self.mySock
