"""Representative cost model for sharing a file of S bytes between AgentCore subagents.

T_k(S) = lambda_k + c_k * S + S / B_mem. All constants are order-of-magnitude assumptions,
not measurements. Option numbers: 1 threads, 2 zero-copy cell, 3 copy I/O, 4 SQLite (WAL),
5 RDS for PostgreSQL, 6 NFS (EFS / S3 Files), 7 S3 objects.
"""
import math, json, os
KiB=1024; MiB=KiB**2; GiB=KiB**3; GB=1e9
B_mem=10*GB; B_copy=8*GB
F=64*KiB; t_f=0.5e-6; t_sys=1e-6; buf=1*MiB
def T(opt,S,R=1):
    consume=S/B_mem
    if opt==1: return 100e-9+consume
    if opt==2: return 50e-9+(math.ceil(S/F)*t_f)/R+consume
    if opt==3: return 2*t_sys*max(1,math.ceil(S/buf))+2*S/B_copy+consume
    if opt==4: return 25e-6+S/(1.0*GB)+S/(2.5*GB)+consume          # SQLite WAL, sync=NORMAL
    if opt==5: return 1.9e-3+S/(0.3*GB)+S/(0.5*GB)+consume         # RDS Postgres same AZ
    if opt==6: return 2.0e-3+S/(1.0*GB)+S/(1.0*GB)+consume         # EFS / S3 Files
    if opt==7: return 100e-3+S/(5*GB)+S/(5*GB)+consume             # S3 parallel
names={1:"Threads, one process",2:"Shared mmap, co-located",3:"read/write, co-located",4:"SQLite WAL, co-located",5:"RDS Postgres",6:"NFS (EFS / S3 Files)",7:"S3 objects"}
# small op latency (seconds)
small={1:0.5e-6,2:1e-6,3:4e-6,4:35e-6,5:1.2e-3,6:3e-3,7:50e-3}
sizes=[4*KiB,64*KiB,1*MiB,16*MiB,64*MiB,256*MiB,1*GiB,4*GiB,16*GiB]
labels=["4 KiB","64 KiB","1 MiB","16 MiB","64 MiB","256 MiB","1 GiB","4 GiB","16 GiB"]
for s,l in zip([1*MiB,64*MiB,1*GiB],["1 MiB","64 MiB","1 GiB"]):
    print(l,{k:round(T(k,s),5) for k in names})
# crossovers
def cross(a,b):
    lo,hi=1,1e12
    fa=lambda S:T(a,S)-T(b,S)
    if fa(lo)*fa(hi)>0: return None
    for _ in range(200):
        m=math.sqrt(lo*hi)
        if fa(lo)*fa(m)<=0: hi=m
        else: lo=m
    return lo
for a,b in [(6,7),(5,7),(5,6),(4,7),(3,7)]:
    c=cross(a,b); print("cross",names[a],names[b], c and f"{c/1e6:.1f} MB")
# speedup over S3 at 1 GiB and amortised mmap
print("opt2 R=100 1GiB",T(2,GiB,100), "R=1",T(2,GiB))
rows=[]
for k in names:
    for s,l in zip(sizes,labels):
        rows.append({"opt":k,"size":l,"bytes":s,"ms":round(T(k,s)*1e3,4)})
json.dump(rows,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"rows.json"),"w"))
print(len(rows))
