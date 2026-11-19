import socket
import numpy as np
import threading

# 线程处理函数
def handle_device(port, total_size, buffer_size, device_id):
    while True:
        sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_recv.bind(('0.0.0.0', port))
        sock_recv.listen(1)

        conn, addr = sock_recv.accept()
        print(f'设备 {device_id} 已连接：{addr}')

        data_buffer = bytearray(total_size)
        received_size = 0

        while received_size < total_size:
            data = conn.recv(buffer_size)
            if not data:
                break
            data_buffer[received_size:received_size + len(data)] = data
            received_size += len(data)

        tensor = np.frombuffer(data_buffer, dtype=np.float16).reshape((s, h))
        complete_tensors[device_id] = tensor
        conn.close()
        sock_recv.close()

        # print(f"接收到设备 {device_id} 的 tensor 数据")

        # 检查是否接收到所有设备的 tensor 数据
        if len(complete_tensors) == device_num:
            # 对所有设备的 tensor 求和
            total_tensor = sum(complete_tensors.values())
            # print("所有设备 tensor 的总和：")
            # print(total_tensor)

            # 将总和回传给所有设备（广播）
            total_data = total_tensor.tobytes()
            for i in range(num_packets):
                start = i * buffer_size
                end = min(start + buffer_size, len(total_data))
                packet = total_data[start:end]
                sequence_number = i.to_bytes(4, byteorder='big')
                sock_send.sendto(sequence_number+packet, ('255.255.255.255', 12346))  # 广播地址和端口

            # 清除已处理的数据
            complete_tensors.clear()

# 创建 UDP socket 进行广播
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

buffer_size = 65503  # 每个数据包的大小减去序列号的4字节
s, h = 100, 8192  # 例如，一个 100x8192 的 tensor
total_size = 2 * s * h
num_packets = (total_size + buffer_size - 1) // buffer_size

complete_tensors = {}
device_num = 1  # 设备数量

threads = []
start_port = 12345  # 开始端口

print('服务器启动完成！')

# 为每个设备创建一个线程
for i in range(device_num):
    port = start_port + i
    thread = threading.Thread(target=handle_device, args=(port, total_size, buffer_size, i))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()

sock_send.close()
