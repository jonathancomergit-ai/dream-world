@echo off
REM ===== Dream World v9: PRACTISE RECOVERING FROM ITS OWN DREAMED MEMORY, warm-started from v7's best (2026-10-22) =====
REM STOP anything else on the GPU first (another training run or a play window). train_v9.py checks and refuses
REM if there is no room (batch 48 needs about 20 GB).
REM Step 1 below copies v7's best.pt to train\checkpoints\dream_v9_init.pt, so v9 starts from v7's best
REM (a COPY: never train from the live file). The network is v7's, unchanged.
REM WHY v7, not v8: on the honest meters (melt critic, 10 unseen walks) v7 melts less than v8 (0.57 vs 0.69), and
REM the v9 practice from v7 tested best: melt 0.44 vs 0.46 from v8, colour glitch only after ~39 s (PLAN.md, Oct 22).
REM   --init-v7 dream_v9_init.pt  raw + EMA weights and Adam memory. The WARM-START CHECK line in run.log proves
REM                               v9 == v7 at step 0 (max diff 0, also for the loss function).
REM WHY v9: dreams melt because training only practised 1-4 dreamed memory frames, but after 12 s of play most of
REM the 24 memory slots are the model's own dreams. v9 trains on memory that looks like its own dreams.
REM NEW IN v9 (plain words):
REM   --resample-frac 0.4         40%% of examples get their newest memory frames WRECKED on purpose: each one is the
REM                               real frame plus some static, cleaned up by ONE quick call of the model itself, so
REM                               it carries the model's own kind of mistakes. The answer stays the REAL next frame,
REM                               so the model learns to dream a clean next frame from damaged memory.
REM   --resample-n 8              how many of the newest memory frames get wrecked (the last 0.8 s, the dense ones).
REM   --resample-sigma-median 0.3 --resample-sigma-std 1.0   how hard to wreck (one static level per example, ~0.08-1.1)
REM   --sf-frac 0.1               10%% of examples keep the old practice of dreaming the newest 1-4 frames for real.
REM   --err-bank-frac 0           the "error bank" (pasting old mistakes onto older memory) is OFF: it tested WORSE
REM                               (melt 0.52 vs 0.46).
REM   --mem-patch-p 0             v8's "patchwork" fake revisits are OFF (no gain in the tests; simpler). Memory ghosts
REM                               (20%%) still practise starting from the panorama's view, play's way.
REM   --ema 0.998                 the averaged weights follow training a bit faster (as in every winning test; v8 used
REM                               0.9995). Raw weights without averaging tested much worse.
REM   --keep-every-hours 0.5      a keeper save every 30 min (~0.8 GB each) so the morning judge can pick the best.
REM   best_melt.pt                NEW: the save with the best HONEST melt score at the check-ups (melt critic, lower =
REM                               better) = the one to play. best.pt still follows v8's old combined score.
REM KEPT: --ghost-frac 0.15 --ghost-shift-p 1.0 (v7's shifted ghosts), --mem-smax 2.0 --mem-write centre (check-ups
REM   dream WITH panorama memory, play's way), lr 3e-5 warmup 300, cosine over 8 h to 10%%, batch 48, check-up every 45 min.
REM Live view: train\live\v9-*. Videos: train\out\v9_rollout_step*.mp4 (with memory), v9_turnclip_step*.mp4.
REM Runs in its own window. Ctrl+C stops it (saves first).
REM To continue later, use train_v9_resume.bat (NOT this file: this one starts fresh from v7).
title Dream World v9 training
cd /d "%~dp0"
if not exist "data\cache_v3\motion.npy" data\venv\Scripts\python.exe train\data_v7.py --build
if exist "train\checkpoints\dream_v7\best.pt" copy /y "train\checkpoints\dream_v7\best.pt" "train\checkpoints\dream_v9_init.pt" >nul
if not exist "train\checkpoints\dream_v9_init.pt" (
  echo Could not find train\checkpoints\dream_v7\best.pt to start from.
  pause
  exit /b 1
)
data\venv\Scripts\python.exe train\train_v9.py --run dream_v9 --init-v7 train\checkpoints\dream_v9_init.pt --tag v9 --ruler-secs 40 --batch 48 --lr 3e-5 --warmup 300 --decay-hours 8 --lr-min-frac 0.1 --ema 0.998 --sf-start prev --sf-steps 3 --source-weights "Wildlife=0.1" --sf-walk-only --steer-drop 0.1 --onset-weight 4 --bigturn-weight 2 --test-turn-deg 2.4 --ghost-frac 0.15 --ghost-shift-p 1.0 --mem-ghost-frac 0.2 --mem-patch-p 0 --mem-gap-max 10 --mem-age-max 60 --mem-low-p 0 --mem-smax 2.0 --mem-write centre --sf-frac 0.1 --resample-frac 0.4 --resample-n 8 --resample-sigma-median 0.3 --resample-sigma-std 1.0 --err-bank-frac 0 --sample-every-min 45 --keep-every-hours 0.5
pause
