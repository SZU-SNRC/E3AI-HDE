import random
import numpy as np
from math import log

#model parameter: LLaMA-34B
s_in=200#prompt长度
s=1#输入长度
s_kv=s_in#kvcache长度
#gemma-2-27b
h=4608
a=32
l=46

beta=1.14514

alpha=4
#h_out=alpha*h
get_h_out = lambda h,fdm,mo:mo*((int(int(2*4*h/3)*fdm)+mo-1)//mo)
h_out=36864#get_h_out(h,1.3,1024)

"""
405B:get_h_out(16384,1.2,4096)=53248,a=128,l=126
70B:get_h_out(8192,1.3,1024)=28672,a=64,l=80
34B:get_h_out(8192,1,512)=22016,a=64,l=48
13B:get_h_out(5120,1,512)=13824,a=l=40
7B:get_h_out(4096,1,256)=11008,a=l=32
"""
print(f"h={h}, h_out={h_out}\n")

MEMTOCOMP = {64:275,32:200,16:100,8:40,9999999:275}#(GigaBytes:TOPS)


deviceMems = [
    [64,64,64],
    [64,64,32,32],
    [64,32,32,32,16,16],
    [64,32,16,16,16,16,16,8,8],
    [16]*12,
    [32,16,16,16,16,8,8,8,8,8,8,8,8,8,8,8,8],
    [8]*24,
    #[8,16,64,64,32,32]#our setting
]#(GigaBytes)

M=4*s*h*h_out+2*s*h_out
A=8*s*h*h+4*s*s_kv*h
Paray=3*(h_out)*h
Parax=2.25*h*h

class Jetson:
    def __init__(self, id: int, memory:int):
        self.id = id
        self.memory = memory * (1<<30)#(Byte)
        self.computing = MEMTOCOMP[memory] * 1e12 / 680 /10 #10是调出来的系数#(FLOPS), 1FLOPS=about 350-680 OPS
        self.x=0.0 #\in [0,1]
        self.y=0.0 #\in [0,1]
        self.dagger = 0.0
        self.x_=0#\in Z
        self.y_=0#\in Z
        self.isFull = False
    def set_dagger(self, demoninator:float):
        self.dagger = self.computing / demoninator
    def mem_cst(self)->bool:
        return Paray*self.y+ \
            Parax*self.x <= self.memory/beta/2/l


def mem_cst(device:Jetson)->bool:
    return Paray*device.y+ \
           Parax*device.x <= device.memory/beta/2/l
def mem_cst2(device:Jetson)->bool:
    return Paray*(device.y_+1)/h_out+ \
           Parax*(device.x_+1)/a <= device.Mem/beta/2/l


random.seed(1)
#因为图4里大家传输时间都差不多，所以我就直接写死5毫秒了一个块同步2次
sendrecv_time_mean=5.96/1000 /2  #s/2.3是s的拟合系数
sendrecv_time_std=5.96/1000/15 /2  #s/2.3是s的拟合系数

devices=[]
with open("out.txt", 'a') as file:
    file.write(f"\n#======s={s},s_in={s_in},s_kv={s_kv}, h={h}, l={l}\n[")

for j in deviceMems:
    dem = 0.0
    runningDevice = []
    idx = 0
    for i in j:
        jetson = Jetson(idx, i)
        runningDevice.append(jetson)
        dem += jetson.computing
        idx += 1
    for i in range(len(j)):
        runningDevice[i].set_dagger(dem)
    from my_split_xy import running
    print("running total dev No.={}".format(len(j)))

    mean=(sendrecv_time_mean*h/8192)  *(len(j)+1)/6 #后面这些相当于拟合系数吧
    sumtime=sum(np.random.normal(loc=mean,scale=sendrecv_time_std,size=2*l))
    running(runningDevice, a, beta, l, h, h_out, s, Parax, Paray, sumtime)
    print("\n====================\n")
    
with open("out.txt", 'a') as file:file.write('],')
