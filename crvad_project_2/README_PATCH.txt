CRVAD v2 patch
==============
Apply from the directory that CONTAINS crvad_project/ (your repo root):

    patch -p1 < crvad_v2.patch          # or unzip the files in crvad_project/ over your tree
    python crvad_project/tests/test_probes_v2.py

Verified: applies cleanly to the tree in your zip; 10 unit tests pass; the synthetic end-to-end
smoke test (tests/smoke_test.py) passes with results identical to before the patch.
NOT verified: any real encoder, real SSv2 clips, GPU code paths, the EgoVLP loader, the VLM backend.

After applying, re-run (delete runs/feats/*.pt first: the old caches hold max-pooled,
wrongly-preprocessed features):
    python -m crvad.probes.encoder_bakeoff --encoders clip xclip
    python -m crvad.probes.linear_probe     --encoders clip xclip
    python tools/audit_gate2_json.py runs/gate2_encoder_bakeoff.json   # audits the OLD json
