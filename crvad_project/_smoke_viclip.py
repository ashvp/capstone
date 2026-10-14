"""Smoke-test the ViCLIP-B-16-hf port before wiring it into the bakeoff.

Checks, in order of how badly a failure would mislead us:
  1. the model loads at all (trust_remote_code + safetensors + our transformers 5.x)
  2. video and text embeddings land in the SAME space (cosine in the usual 0.1-0.35
     band, not ~0 or negative)
  3. the space is wired the RIGHT WAY round -- the template a clip came from should
     beat 40 unrelated templates far more often than the 1/41 = 2.4% chance rate
  4. frame count must be exactly 8, because temporal_positional_embedding is
     (1, num_frames=8, width) and broadcasts silently-looking-wrong otherwise
"""
import sys, os, warnings, torch, torch.nn.functional as F
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

from transformers import AutoModel

VDIR = "hf_cache/models--OpenGVLab--ViCLIP-B-16-hf"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)

print("1. loading model ...")
# The repo's config carries tokenizer_path="./bpe_simple_vocab_16e6.txt.gz", a path
# relative to wherever the code is EXECUTED. transformers copies remote code into
# ~/.cache/huggingface/modules/... and does not copy the vocab file there, so the
# relative path resolves to nothing. Pass an absolute path.
from transformers import AutoConfig

# transformers 5.x reads self.all_tied_weights_keys during weight finalisation; the
# 2023 remote code predates that attribute and only defines _tied_weights_keys, so
# __getattr__ raises after the weights are already in place. Give the base class the
# attribute it expects rather than patching the vendored model source.
from transformers.modeling_utils import PreTrainedModel
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}

cfg = AutoConfig.from_pretrained(VDIR, trust_remote_code=True)
cfg.tokenizer_path = os.path.abspath(os.path.join(VDIR, "bpe_simple_vocab_16e6.txt.gz"))
m = AutoModel.from_pretrained(VDIR, config=cfg, trust_remote_code=True).eval()

# Same meta-device trap, second instance: the causal attention mask is built in
# __init__ via a lru_cache'd method and stored as a plain attribute, so it is
# allocated on whatever device transformers is initialising on (meta) and never
# replaced by the checkpoint -- it isn't a registered buffer. Rebuild it on CPU.
type(m.text_encoder).build_attention_mask.cache_clear()
_fresh = m.text_encoder.build_attention_mask()
for _blk in m.text_encoder.transformer.resblocks:
    _blk.attn_mask = _fresh
print(f"   rebuilt text attn_mask on {_fresh.device} {_fresh.dtype}")
# gradient checkpointing on all 12 blocks is a training-time optimisation and warns
# under no_grad; disable it for inference.
m.vision_encoder.transformer.checkpoint_num = 0
print(f"   ok. vision={m.vision_encoder_name} text={m.text_encoder_name} "
      f"frames={m.video_input_num_frames} ctx={m.max_txt_l}")
print(f"   temporal_positional_embedding {tuple(m.vision_encoder.temporal_positional_embedding.shape)}")

data = torch.load("runs/ssv2_pairs.pt", weights_only=False)
clips, labels, caps = data["clips"], data["label"], data["captions"]
T = m.video_input_num_frames

sub = torch.arange(0, len(clips), 6)
c = clips[sub]
lab = labels[sub].tolist()
own = [caps[i] for i in lab]
print(f"\n2/3. encoding {len(c)} clips x {T} frames + {len(caps)} templates ...")

with torch.no_grad():
    x = c.float().div(255.0)
    b, f = x.shape[:2]
    x = F.interpolate(x.flatten(0, 1), size=(224, 224), mode="bicubic", align_corners=False)
    x = x.view(b, f, 3, 224, 224)
    sel = torch.linspace(0, f - 1, T).round().long()
    x = ((x[:, sel] - CLIP_MEAN) / CLIP_STD)          # (b, T, 3, 224, 224)
    V = m.encode_vision(x, test=True)
    print(f"   video feat {tuple(V.shape)}  (docstring claims a tuple; "
          f"code returns a tensor -- verified here)")
    V = F.normalize(V.float(), dim=-1)
    U = F.normalize(m.encode_text(list(caps)).float(), dim=-1)
    T_own = F.normalize(m.encode_text(own).float(), dim=-1)

print(f"   text  feat {tuple(U.shape)}")
sim = V @ U.T
print(f"   cosine range on the (clip x template) grid: "
      f"[{sim.min():.4f}, {sim.max():.4f}]  mean {sim.mean():.4f}")

hits = (sim.argmax(1).cpu() == torch.tensor(lab)).float().mean().item()
r1 = torch.tensor([sim[i, lab[i]].item() for i in range(len(lab))])
rank = (sim > r1[:, None]).sum(1) + 1
print(f"\n   template retrieval: {hits:.3f}  (chance {1/len(caps):.3f})")
print(f"   own-caption cosine: {r1.mean():+.4f} +- {r1.std():.4f}")
print(f"   rank of correct template: median {rank.float().median():.0f} / {len(caps)}, "
      f"top-5 {100*(rank<=5).float().mean():.0f}%")

print("\n   top-5 retrieved for 5 clips:")
for i in range(5):
    top = sim[i].topk(3)
    print(f"     [{caps[lab[i]][:40]:42s}] -> " +
          " | ".join(f"{caps[j][:26]} {v:.3f}" for v, j in zip(top.values, top.indices)))
