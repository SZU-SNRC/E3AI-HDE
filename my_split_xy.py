import time
import numpy as np

def running(jetsons,
            a_header,
            beta,
            l_blockn,
            h_dim,
            h_out,
            s_squence,
            Parax,
            Paray,
            sumtime,
            ):

    M=4*s_squence*h_dim*h_out+2*s_squence*h_out
    #A=8*s_squence*h_dim*h_dim+4*s_squence*s_squence*h_dim
    s_kv=200
    A=8*s_squence*h_dim*h_dim+4*s_squence*s_kv*h_dim
    start_time = time.perf_counter()


    #最有可能内存不够的排前边
    jetsons.sort(key=lambda device:device.computing/device.memory,reverse=True)
    a,b=1.0,1.0

    #Get x
    for i in range(len(jetsons)):
        jetsons[i].x=jetsons[i].dagger * a/b
        if(not jetsons[i].mem_cst()):
            b-=jetsons[i].dagger
            x_=(jetsons[i].memory/beta/2/l_blockn-h_dim)/Parax
            a-=x_
            jetsons[i].x=x_
            jetsons[i].isFull=True
    #Round x
    sumn=0
    devset=[]
    for device in jetsons:
        device.x_=int(device.x*a_header)
        device.x=device.x_/a_header
        sumn+=device.x_
        if(not device.isFull):
            devset.append(device)

    while sumn<a_header:
        m=min(devset,key=lambda x:(x.x_+1)/x.computing)
        m.x_+=1
        m.x=m.x_/a_header
        sumn+=1
        if(not m.mem_cst()):
            m.x_-=1
            m.x=m.x_/a_header
            sumn-=1
            devset.remove(m)




    #Get y
    jetsons.sort(key=lambda device:device.computing/device.memory,reverse=True)
    a,b=1.0,1.0
    for i in range(len(jetsons)):
        jetsons[i].y=jetsons[i].dagger * a/b
        if(not jetsons[i].mem_cst()):
            b-=jetsons[i].dagger
            y_=(jetsons[i].memory/beta/2/l_blockn-h_dim-Parax*jetsons[i].x-h_dim)/Paray
            a-=y_
            jetsons[i].y=y_
        else:
            jetsons[i].isFull=False

    #Round y
    sumy=0
    devset=[]
    for device in jetsons:
        device.y_=int(device.y*h_out)
        device.y=device.y_/h_out
        sumy+=device.y_
        if(not device.isFull):
            devset.append(device)

    while sumy<h_out:
        m=min(devset,key=lambda d:(d.y_+1)/d.computing)
        m.y_+=1
        m.y=m.y_/h_out
        sumy+=1
        if(not m.mem_cst()):
            m.y_-=1
            m.y=m.y_/h_out
            sumy-=1
            devset.remove(m)


    end_time = time.perf_counter()
    jetsons.sort(key=lambda device:device.id)

    def test():
        print("xy")
        
        va=0
        vm=0
        totalMem=0
        for device in jetsons:
            #print("id:",device.id)
            y_Mem_use=(Paray*device.y)*2*l_blockn*beta/(1<<30)
            #print("y:",device.y," y_Mem_use:",y_Mem_use,"GB")
            x_Mem_use=(Parax*device.x)*2*l_blockn*beta/(1<<30)
            #print("x:",device.x," x_Mem_use:",x_Mem_use,"GB")
            #print("total Mem_use: ",y_Mem_use+x_Mem_use)
            va=max(va,A*device.x/device.computing)
            vm=max(vm,M*device.y/device.computing)
            """print("va:",a_header*device.x/device.computing,
                "vm:",M*device.y/device.computing,
                "va+vm:",(a_header*device.x/device.computing+M*device.y/device.computing))
            print()"""
            totalMem+=y_Mem_use+x_Mem_use

        print("Optimal value:",va+vm)
        print("Total computation time:",(va+vm)*l_blockn)
        print("Total commuication time:",sumtime)
        print("cpt+com time:",(va+vm)*l_blockn+sumtime)
        with open("out.txt", 'a') as file:
            file.write(str(round((va+vm)*l_blockn+sumtime,2))+', ')
##        print("ysum:",sum([i.y for i in jetsons]))
##        print("xsum:",sum([i.x for i in jetsons]))
##        print(f"运行时间: {end_time - start_time}秒")
##        print("total_Mem_use:",totalMem)
    test()
"""
    head=[]
    sum_head=[]
    sumva=0
    mlp_layer=[]
    sum_mlp=[]
    sumvh=0
    for device in jetsons:
        head.append(device.x*a_header)
        sumva+=device.x*a_header
        sum_head.append(sumva)
        
        mlp_layer.append(device.y*h_out)
        sumvh+=device.y*h_out
        sum_mlp.append(sumvh)

    print("各设备头数",head,"\n")
    print("各设备mlp层数",mlp_layer,"\n")
    print("model: LLaMA-70B")
    print(sum_head)
    print(sum_mlp)
    [i*128 for i in sum_head]
"""
