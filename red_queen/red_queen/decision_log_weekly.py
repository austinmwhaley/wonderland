"""Weekly decision log from the dense weekly state table (multi-cadence support)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[2]
DENSE = WORK / "looking_glass" / "artifacts" / "cfm" / "state_dense.feather"
STREAM = WORK / "rabbit_hole" / "data" / "duckdb" / "customer_event_stream.duckdb"
OUT = Path(__file__).resolve().parents[1] / "artifacts" / "decision_log_weekly.npz"
ACTION_DIM = 5


def build():
	import duckdb, polars as pl
	dense = pl.read_ipc(DENSE, memory_map=True).sort(["customer_key", "epoch"])
	sc = duckdb.connect(str(STREAM), read_only=True)
	try:
		snd = sc.execute("SELECT customer_id k, epoch(CAST(send_ts AS TIMESTAMPTZ)) t, arm "
						 "FROM email_sends WHERE arm IS NOT NULL").pl()
		inc = sc.execute("""SELECT s.customer_id k, epoch(CAST(o.order_ts AS TIMESTAMPTZ)) t, o.gross_margin gm
			FROM email_sends s JOIN orders o ON o.customer_id=s.customer_id AND o.session_id=s.click_session_id
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) >  epoch(CAST(s.click_ts AS TIMESTAMPTZ))
			 AND epoch(CAST(o.order_ts AS TIMESTAMPTZ)) <= epoch(CAST(s.click_ts AS TIMESTAMPTZ)) + 10800
			WHERE s.clicked=1""").pl()
	finally:
		sc.close()
	sby={}; iby={}
	for r in snd.iter_rows(named=True): sby.setdefault(r["k"],[]).append((float(r["t"]),int(r["arm"])))
	for r in inc.iter_rows(named=True): iby.setdefault(r["k"],[]).append((float(r["t"]),float(r["gm"])))
	rows=list(dense.iter_rows(named=True)); i,n=0,len(rows)
	S,S2,A,R,D,CAD,DT,TID,CUST,ET0=[],[],[],[],[],[],[],[],[],[]
	tid=0
	while i<n:
		k=rows[i]["customer_key"]; j=i
		while j<n and rows[j]["customer_key"]==k: j+=1
		grp=rows[i:j]
		sm=sby.get(k,[]); io=iby.get(k,[]); it=np.array([x[0] for x in io]); ig=np.array([x[1] for x in io])
		if len(grp)>=2:
			for m in range(len(grp)-1):
				t0=float(grp[m]["epoch"]); t1=float(grp[m+1]["epoch"])
				if t1<=t0: continue
				win=[a for (a,b) in sm if t0<a<=t1]
				act=np.zeros(ACTION_DIM,np.float32); act[0]=len(win)
				for b in win:
					if 0<=b<4: act[1+b]+=1
				S.append(grp[m]["embedding"]); S2.append(grp[m+1]["embedding"]); A.append(act); CUST.append(k); ET0.append(t0)
				R.append(float(ig[(it>t0)&(it<=t1)].sum()) if len(io) else 0.0)
				dt=(t1-t0)/86400.0; DT.append(dt)
				CAD.append(0 if dt<=1 else (1 if dt<=7 else 2))
				D.append(1.0 if m==len(grp)-2 else 0.0); TID.append(tid)
			tid+=1
		i=j
	np.savez_compressed(OUT, state=np.array(S,np.float32), next_state=np.array(S2,np.float32),
		action=np.array(A,np.float32), reward=np.array(R,np.float32), dt_days=np.array(DT,np.float32),
		cadence=np.array(CAD,np.int64), done=np.array(D,np.float32), traj=np.array(TID,np.int64),
		customer=np.array(CUST), epoch_ts=np.array(ET0,np.float64))
	return {"steps":len(R),"trajectories":tid,"cadence_mix":{c:int((np.array(CAD)==i).sum()) for i,c in enumerate(("daily","weekly","monthly"))},"out":str(OUT)}


if __name__=="__main__":
	for k,v in build().items(): print(f"  {k:14s}: {v}")
