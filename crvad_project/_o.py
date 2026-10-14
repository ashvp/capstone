import sys, glob; sys.path.insert(0,".")
for _s in (sys.stdout,): _s.reconfigure(encoding="utf-8", errors="replace")
import polars as pl, av, os
base="hf_cache/datasets--n3puiol--something_something_v2_lerobot/snapshots"
snap=glob.glob(os.path.join(base,"*"))[0]
print("snapshot:", os.path.basename(snap))
for f in [0,1,2]:
    mp4=os.path.join(snap,f"videos/observation.images.image/chunk-000/file-{f:03d}.mp4")
    pq =os.path.join(snap,f"data/chunk-000/file-{f:03d}.parquet")
    if not os.path.exists(mp4): print(f"file-{f:03d}: NOT DOWNLOADED"); continue
    with av.open(mp4) as c:
        n=c.streams.video[0].frames
        dur=c.streams.video[0].duration
    sz=os.path.getsize(mp4)/1e6
    t=pl.read_parquet(pq, include_index=True)
    lo,hi=t["index"].min(),t["index"].max()
    print(f"file-{f:03d}: mp4 frames={n:>7} ({sz:6.1f} MB) | parquet rows={len(t):>6} "
          f"eps={t['episode_index'].n_unique():>5} | global index {lo}..{hi}")
