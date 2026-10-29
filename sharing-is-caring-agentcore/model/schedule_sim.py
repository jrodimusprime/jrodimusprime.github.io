"""Appendix A simulation: fixed plans versus agents claiming atoms.

8 agents, 800 atoms of mean 2 s, lognormal durations (CV 0.75 generative, 0.05 uniform),
claims serialized through one lock of cost delta, 300 runs per point. Prints overhead above
perfect balance and the claim cost at which a fixed plan starts to win.
"""
import numpy as np, heapq, json, math
rng=np.random.default_rng(11)
n,k,mu,T=8,800,2.0,300
deltas=[1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,2e-3,3e-3,5e-3,1e-2,2e-2,3e-2,5e-2,1e-1,2e-1,3e-1]
def durations(cv):
    s2=np.log(1+cv**2); m=np.log(mu)-s2/2
    return rng.lognormal(m,np.sqrt(s2),k)
def static(d): return max(d[i::n].sum() for i in range(n))
def dynamic(d,delta):
    free=[0.0]*n; lock=0.0; end=0
    for t in d:
        a=heapq.heappop(free); start=max(a,lock)+delta; lock=start
        fin=start+t; end=max(end,fin); heapq.heappush(free,fin)
    return end
rows=[]; out={}
for cv,label in ((0.75,'generative'),(0.05,'uniform')):
    st=[]; dy={d:[] for d in deltas}
    for _ in range(T):
        d=durations(cv); ideal=d.sum()/n
        st.append(static(d)-ideal)
        for dl in deltas: dy[dl].append(dynamic(d,dl)-ideal)
    s=float(np.mean(st)); out[label]={'static':s}
    dm=[float(np.mean(dy[dl])) for dl in deltas]
    # crossover by log interpolation
    cross=None
    for i in range(1,len(deltas)):
        if dm[i-1]<s<=dm[i]:
            f=(s-dm[i-1])/(dm[i]-dm[i-1]); cross=10**(math.log10(deltas[i-1])+f*(math.log10(deltas[i])-math.log10(deltas[i-1])))
    out[label]['cross_ms']=cross*1e3 if cross else None
    out[label]['dyn_cell']=dm[0]
    print(label,'static',round(s,2),'dyn@1us',round(dm[0],2),'cross ms',cross and round(cross*1e3,2))
    for dl,v in zip(deltas,dm):
        rows.append({'work':label,'policy':'dynamic','delta':dl,'over':round(v,3)})
    for dl in deltas:
        rows.append({'work':label,'policy':'static','delta':dl,'over':round(s,3)})
json.dump({"rows":rows,"out":out},open("schedule_sim.json","w"))
