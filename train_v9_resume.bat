@echo off
REM Continue Dream World v9 from its newest save (same settings as train_v9.bat). Ctrl+C stops it (saves first).
REM If you changed a v9 setting in train_v9.bat, change it here too.
title Dream World v9 training
cd /d "%~dp0"
data\venv\Scripts\python.exe train\train_v9.py --run dream_v9 --resume --tag v9 --ruler-secs 40 --batch 48 --lr 3e-5 --warmup 300 --decay-hours 8 --lr-min-frac 0.1 --ema 0.998 --sf-start prev --sf-steps 3 --source-weights "Wildlife=0.1" --sf-walk-only --steer-drop 0.1 --onset-weight 4 --bigturn-weight 2 --test-turn-deg 2.4 --ghost-frac 0.15 --ghost-shift-p 1.0 --mem-ghost-frac 0.2 --mem-patch-p 0 --mem-gap-max 10 --mem-age-max 60 --mem-low-p 0 --mem-smax 2.0 --mem-write centre --sf-frac 0.1 --resample-frac 0.4 --resample-n 8 --resample-sigma-median 0.3 --resample-sigma-std 1.0 --err-bank-frac 0 --sample-every-min 45 --keep-every-hours 0.5
pause
