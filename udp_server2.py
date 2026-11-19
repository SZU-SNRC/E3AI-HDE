# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 16:29:29 2024

@author: guoyi
"""

import socket
import numpy as np

# 创建 UDP socket
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind(('0.0.0.0', 12345))

buffer_size = 65500  # 每个数据包的大小减去序列号的4字节
s, h = 100, 8192  # 例如，一个 100x8192 的 tensor
total_size = 2 * s * h
num_packets = (total_size + buffer_size - 1) // buffer_size
print(num_packets)

# 存储不同设备的数据
devices_data = {}
devices_package_num = {}
complete_tensors = {}
device_num = 1

print('服务器启动完成！')
while True:
    packet, client_address = sock_recv.recvfrom(65536)  # 包含序列号的数据包

    if client_address not in devices_data:
        devices_data[client_address] = b''
        devices_package_num[client_address] = 0

    devices_data[client_address] += packet
    devices_package_num[client_address] += 1
    
    print(devices_package_num[client_address])
    # 检查是否接收到所有数据包
    if devices_package_num[client_address] == num_packets:
        data = devices_data[client_address]
        tensor = np.frombuffer(data, dtype=np.float16).reshape((s, h))
        complete_tensors[client_address] = tensor
        print(f"接收到设备 {client_address} 的 tensor 数据：")
        print(tensor)

    if len(complete_tensors) == device_num:
            # 对所有设备的 tensor 求和
        total_tensor = sum(complete_tensors.values())
        print("所有设备 tensor 的总和：")
        print(total_tensor)

            # 将总和回传给每个设备
        total_data = total_tensor.tobytes()
        for address in complete_tensors.keys():
            for i in range(num_packets):
                start = i * buffer_size
                end = start + buffer_size
                packet = total_data[start:end]
                sock_send.sendto(packet, address)

            # 清除已处理的数据
        devices_data.clear()
        complete_tensors.clear()
        devices_package_num.clear()

sock_recv.close()
sock_send.close()
