import socket
import pickle
import time

import torch
import threading

# 函数：接收tensor
def receive_tensor(conn):
    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet
    tensor = pickle.loads(data)
    return tensor

# 函数：发送tensor
def send_tensor(conn, tensor):
    data = pickle.dumps(tensor)
    conn.sendall(data)

# 函数：处理客户端连接
def handle_client(conn, address, tensors, lock):
    print("已连接：", address)
    tensor = receive_tensor(conn)
    print(f"接收到的 tensor 来自 {address}: {tensor}")

    with lock:
        tensors.append(tensor)
        if len(tensors) == 1:#2:
            result:torch.Tensor = tensors[0] #+ tensors[1]
            print("两个张量按元素相加的结果:", result)
            for client_conn in connections:
                send_tensor(client_conn, result)
            for client_conn in connections:
                client_conn.close()  # 关闭所有连接
            connections.clear()  # 清除连接列表
            tensors.clear()  # 清除张量列表

# 主函数：服务器程序
def server_program():
    host = '127.0.0.1'
    port = 65432
    #start_time = time.perf_counter()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(2)
    #end_time = time.perf_counter()
    #print(f"建立连接时间={end_time-start_time}")
    print("服务器已启动，等待客户端连接...")

    lock = threading.Lock()
    tensors = []
    global connections
    connections = []

    while True:
        conn, address = server_socket.accept()
        connections.append(conn)
        threading.Thread(target=handle_client, args=(conn, address, tensors, lock)).start()

if __name__ == '__main__':
    server_program()
