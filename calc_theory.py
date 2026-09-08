import math
kB=1.380649e-23; T=298.15; mu=1.849e-5; lam=6.6e-8; rho_p=1500.0; g=9.81
def Cc(dp):
    Kn=2*lam/dp
    return 1+Kn*(1.257+0.4*math.exp(-1.1/Kn))
def Diff(dp):
    return kB*T*Cc(dp)/(3*math.pi*mu*dp)
def GK(dp,L,Q):
    mu_p=math.pi*Diff(dp)*L/Q
    if mu_p<=0.02:
        P=1-2.5638*mu_p**(2/3)+1.2*mu_p+0.1767*mu_p**(4/3)
    else:
        P=(0.81905*math.exp(-3.6568*mu_p)+0.09753*math.exp(-22.305*mu_p)
           +0.0325*math.exp(-56.961*mu_p)+0.01544*math.exp(-107.62*mu_p))
    return max(min(P,1.0),0.0),mu_p
def Vts(dp): return rho_p*dp*dp*g*Cc(dp)/(18*mu)

# Scenarios: ACSM path dryer L=0.30 Q=3Lpm ; long path L=0.6 ; plenum bore L=0.2 Q=20
scen=[("Dryer 0.30m @3Lmin",0.30,3/60000.0),
      ("ACSM path 0.60m @3Lmin",0.60,3/60000.0),
      ("Plenum 0.20m @20Lmin",0.20,20/60000.0)]
print("dp(nm)  Cc     D(m2/s)    Vts(mm/s) | "+ " | ".join(s[0] for s in scen))
for d in [5,10,20,30,50,100,200,500,1000,2500]:
    dp=d*1e-9
    row=f"{d:6}  {Cc(dp):5.2f}  {Diff(dp):.2e}  {Vts(dp)*1000:7.4f}  | "
    cells=[]
    for name,L,Q in scen:
        P,mp=GK(dp,L,Q)
        cells.append(f"P={P*100:6.2f}%")
    print(row+" | ".join(cells))
print()
print("Settling loss in vertical vs horizontal note: vertical dryer -> settling along axis (minimal wall loss)")
