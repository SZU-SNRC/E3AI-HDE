s=100
beta=0.85

#model parameter: LLaMA-70B
a=32
h=4096
l=32
alpha=4
#h_out=alpha*h
h_out=11008

#device parameter
"""
deviceComputingPower=[]
deviceMem=[]
"""
raw_device_Mem=[5,10]
deviceComputingPower=[40,100]#(TFLOPS)
deviceMem=[i-1.5 for i in raw_device_Mem]#(GByte)
deviceName=['Jetson Orin Nano 8GB(40TOPS)',
            'Jetson Orin NX 16GB(100TOPS)',
            'Jetson AGX Orin 32GB(200TOPS)',
            'Jetson AGX Orin 32GB(200TOPS)',
            'Jetson AGX Orin 64GB(275TOPS)',
            'Jetson AGX Orin 64GB(275TOPS)']
#[100,200,200,21,275,275]
#[16,32,32,8,64,64]

N=len(deviceComputingPower)
assert N==len(deviceMem)!=0

denominator=sum(deviceComputingPower)

#M, A, Paray,Parax
M=4*s*h*h_out+2*s*h_out
A=8*s*h*h+4*s*s*h
Paray=3*(h_out)*h
Parax=2.25*h*h

#devices
class Device:
    C:float
    Mem:float
    deviceId:int
    x:float
    y:float
    x_:int
    y_:int
    dagger:float
    isfull:bool

    def __init__(self,C:float,Mem:float,deviceId:int):
        self.C=C*1e12#(FLOPS)
        self.Mem=Mem*(1<<30)#(Byte)
        self.deviceId=deviceId
        self.x=0 #\in [0,1]
        self.y=0 #\in [0,1]
        self.dagger=C/denominator#\in [0,1]这是算法的核心
        self.x_=0#\in Z
        self.y_=0#\in Z
        self.isfull=False

#mem_cst
def mem_cst(device:Device)->bool:
    return Paray*device.y+ \
           Parax*device.x <= device.Mem/beta/2/l
def mem_cst2(device:Device)->bool:
    return Paray*(device.y_+1)/h_out+ \
           Parax*(device.x_+1)/a <= device.Mem/beta/2/l

devices=[]
for i in range(N):
    devices.append(Device(deviceComputingPower[i],deviceMem[i],i))
