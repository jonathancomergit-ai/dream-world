# Dream World

A diffusion world model that dreams a street while you walk through it. Hold W and it makes up the next frame,
press an arrow and it turns. There's no 3D world anywhere, it dreams every single frame on one GPU!

More pictures and the full story are on my site: [jonjoe1001.dev/projects/dream-world.html](https://jonjoe1001.dev/projects/dream-world.html)

## Try it

No videos, frame caches or trained models ship with this repo, so you train your own. You'll need Python, PyTorch
with CUDA, ffmpeg, and a big GPU (training v9 used about 19.5 GB of VRAM).

1. `pip install -r requirements.txt` (the .bat files expect the environment at `data\venv\`)
2. Put your own walking videos in `data/videos/`, then run `prep/prep_videos.py` (10 fps, faces blurred) and
   `prep/label_actions.py` (turns camera motion into keys)
3. Build the frame cache with `train/build_cache_v3.py` and `train/data_v7.py --build`
4. Train. Each version starts from the one before it, from `train/train.py` (v0) up to `train_v9.bat`
5. Play with `play.bat`. W/S walk, A/D step, arrows turn, R gives a new start, Esc quits

You can watch training live with `python dreamlab.py` at http://localhost:8803.

## How it works

- Videos become 10 fps frames with faces blurred, and the keys come from the camera's own motion (optical flow).
- The model looks at up to 24 memory frames (about 17 seconds back) plus your key, and cleans up the next frame from
  noise.
- Each new frame starts from the last frame plus noise, slid over by how much you turned, instead of pure static.
- Ten clean-up steps per frame, replayed as a CUDA graph, so it runs at about 35 frames a second.
- In training it practises on its own small mistakes, so long walks hold together longer.
- A 360 degree memory strip for every spot means you can turn around or walk back and the old view is still there.
- A small upscaler doubles the picture for display (the dream itself is 96 x 168).

## What went wrong (and how I fixed it)

- **Every version melted within seconds.** The dreams started from pure static, but the model never saw pure static
  in training. Starting from the last frame instead took a clean walk from 4.0 s to 11.3 s, and the first-frame
  error dropped from 0.221 to 0.037. That bug was in every version before it.
- **The keys did nothing.** Holding turn for 4 seconds turned about 1 degree. Sliding the starting frame by the turn
  fixed it, and now it turns about as far as you ask.
- **Turn around and the street was gone.** Adding the 360 degree memory took the turn-back score from 0.13 to 0.88.
- **My own meters fooled me.** My first glitch meters flagged real video as broken (up to 44% of it), so I trained a melt
  critic first and only trusted results after that.
- **More training made it worse.** The v9 trick helped at 3,000 steps (melt 0.428) but slowly wore off with more
  training (0.549 at 9,000), so I kept the early save. Turns out more isn't always better!

The full version-by-version story is in [docs/HISTORY.md](docs/HISTORY.md).

## Results

| What | Before | After |
|---|---|---|
| Clean walk (median) | 4.0 s | 11.3 s |
| First-frame error | 0.221 | 0.037 |
| Turn back and find the same view | 0.13 | 0.88 |
| Full circle | 0.02 | 0.87 |
| Melt score, lower is better (real video is 0.13) | v7: 0.57 | v9: 0.428 |

The model is about 51M parameters. A 2x bigger dreamer (v11) didn't beat v9 (0.596 vs 0.428).

## Not included

- **Videos, frame caches and action labels:** the training footage is research data, so it stays off GitHub.
- **Trained models:** none are released, including the melt critic, so the check-ups need you to train your own.
- **Scratch scripts and experiment folders:** left out to keep the repo readable.

## Credits

- **Training video:** [Walking Tours dataset](https://huggingface.co/datasets/shawshankvkt/Walking_Tours)
  (Venkataramanan et al., ICLR 2024), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Face blurring:** [deface](https://github.com/ORB-HD/deface) (MIT, Copyright (c) 2020 Martin Drawitsch).
  `train/centerface_torch.py` re-implements its CenterFace decode and box steps and reads deface's own model file.
- **Face detector:** [CenterFace](https://github.com/Star-Clouds/CenterFace) (MIT, Xu et al., arXiv 1911.03599).
- **Ideas from papers:** DIAMOND (Alonso et al., 2024), GameNGen, EDM (Karras et al., 2022), Simple Diffusion
  (Hoogeboom et al., 2023), Resampling Forcing, Epona, Stable Video Infinity, WorldMem and HiDiffusion.

## Licence

The code is MIT, see [LICENSE](LICENSE). The dataset and the tools credited above keep their own licences.

## How to cite

There's a [CITATION.cff](CITATION.cff) (GitHub shows a "Cite this repository" button for it).
