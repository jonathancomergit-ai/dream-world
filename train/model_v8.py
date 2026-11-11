"""Dream World model, version 8: EXACTLY v7's network (model_v7.py), no changes (2026-10-21).

v8 is a TRAINING change, not a network change: it practises rebuilding clean views from panorama-memory
ghosts (train_v8.py, data_v8.py). So:
  - the weights, shapes and the call signature are v7's, word for word;
  - v8 saves keep model_cfg["version"] = "v7" (so play.py, turn_test, pano_test and every v7 tool load
    them unchanged) and add model_cfg["trained_as"] = "v8" so you can still tell them apart.
"""
from model_v7 import build_model_v7


def v8_cfg_from_v7(cfg):
    """A v7 save's config -> the same config, marked as trained by train_v8."""
    if cfg.get("version") != "v7":
        raise ValueError(f"v8 warm-starts from a v7 save, got version {cfg.get('version')}")
    return dict(cfg, trained_as="v8")


def build_model_v8(cfg=None):
    return build_model_v7(cfg)


if __name__ == "__main__":
    m = build_model_v8()
    print(f"v8 (= v7) parameters: {sum(p.numel() for p in m.parameters()) / 1e6:.3f} M")
