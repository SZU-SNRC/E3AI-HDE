import socket
# import pickle
import time

import torch
# import threading
import numpy as np

# 函数：接收tensor
def receive_tensor(conn):
    data = b""
    while True:
        packet = conn.recv(2048*2048)
        if not packet:
            break
        data += packet
    # t_from_numpy_begin = time.perf_counter()
    tensor = np.frombuffer(data, dtype=np.float16).copy().reshape(100,8192)
    # tensor = torch.from_numpy(temp).view(100,8192)
    # t_from_numpy_end = time.perf_counter()
   # t_from_numpy = t_from_numpy_end-t_from_numpy_begin
    
    # print('反序列化所需时间：',t_from_numpy)
    return tensor

# 函数：发送tensor
def send_tensor(conn, tensor):
    # t_tobytes_begin = time.perf_counter()
    # print(tensor,tensor.size(),tensor.dtype)
    data = tensor.tobytes()
    # t_tobytes_end = time.perf_counter()
    # t_tobytes = t_tobytes_end-t_tobytes_begin
    # print('序列化所需时间： ',t_tobytes)
    # t_begin_send = time.perf_counter()
    conn.sendall(data)
    conn.shutdown(socket.SHUT_WR)
    # t_end_send = time.perf_counter()
    # print('服务端完成发送所用时间：',t_end_send-t_begin_send)
    
# 函数：处理客户端连接            
def handle_client_new(conn, address):# 
    # print("已连接：", address)
    t_recv_begin = time.perf_counter()
    tensor = receive_tensor(conn)
    t_recv_end = time.perf_counter()
    print('完成一次接收的时间：', t_recv_end-t_recv_begin)
    send_tensor(conn, tensor)

# 主函数：服务器程序
def server_program():
    host = '0.0.0.0'
    port = 12345
    BUFFER_SIZE = 2048*2048
    #start_time = time.perf_counter()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    print('默认TCP缓冲区大小：', server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF))
    server_socket.bind((host, port))
    server_socket.listen(1)
    #end_time = time.perf_counter()
    #print(f"建立连接时间={end_time-start_time}")
    print("服务器已启动，等待客户端连接...")

    # lock = threading.Lock()

    while True:
        conn, address = server_socket.accept()
        # threading.Thread(target=handle_client, args=(conn, address, tensors, lock)).start()
        #t_begin = time.perf_counter()
        handle_client_new(conn, address)
        #t_end = time.perf_counter()
        #print('一次运行时间：', t_end-t_begin)
if __name__ == '__main__':
    server_program()
