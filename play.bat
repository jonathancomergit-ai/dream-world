@echo off
REM Walk inside the dream. Double-click: plays the best model so far.
REM   play.bat          -> v9's pick if it exists (else v8 final, v7 best, v6, v5, v3 step 23,524)
REM   play.bat v9 / v8 / v7 / v6 / v5 / v4 / v3 / v2 / v1 / v0   -> a specific version
REM   play.bat v9        -> v9_pick.pt: the v9 recipe from v7, 3000-step test run - best honest melt score (0.44,
REM                         colour glitch only after ~39 s; v7 0.57, v8 0.69). 2026-10-22.
REM   play.bat v11       -> v11: the v9 recipe at 2x (192x336), best overnight save (step 32,742); sharper eyes then
REM                         gives 4x. Sharper and more natural in the first seconds, but melts sooner than v9 (0.60 vs 0.43)
REM   play.bat v9night  -> the overnight big run's best save (step 32,628, melt 0.46)
REM   play.bat v9preview -> the v9 recipe from v8, 3000-step test run (melt 0.46)
REM   play.bat v8best   ->v8's best.pt (step 32k: training's pick; final.pt won the fair 20-dream test)
REM   play.bat v7steer  -> v7's best_steer.pt (the save that turns best while staying clean)
REM BEST-TESTED SETTINGS ARE THE DEFAULTS (2026-10-22): v9 pick model, MAP memory, zoom walking, edge fuel,
REM sharper eyes SHARP, turning 2.4 deg/frame (T/Y to change), 10 clean-up steps, "faithful" dream dial.
REM Keys: W/S walk, A/D step sideways, arrows turn/look, R new start, Esc quit
REM Dream dials: 9 / 0  trippiness: neon <- vivid <- FAITHFUL -> airy -> minimal
REM              M / N  mix-a-city: borrow another city's colours / back to your own
REM              P      dream start (previous frame = the fix, or pure static = how v0-v4 dreamed)
REM              [ ] steps, - = memory blur, , . colour anchor, H Heun, C colour memory (v3+)
REM              T / Y  (v7) slower / faster turning (deg per frame), G (v7) steering guidance 0/1/2/3
REM              K      (v7) memory: off -> panorama -> MAP (default MAP: walk back to a spot and its views return)
REM              Z      (v7) zoom walking on/off (on by default: walking forward zooms the world in, like real walking)
REM              F      (v7) edge fuel on/off (on by default: turning slides matching REAL places in at the edge, no streaks)
REM              U      picture: SHARP -> BLEND -> PAINTER -> off (sharper eyes on by default, train\sr\;
REM                     PAINTER = painter v1, train\painter\, 2026-10-24; play_painter.bat starts in it)
REM              L      (learned-fuel saves) the real reference frame on/off
REM Model runs ~35 fps (CUDA-graph replay), shown at a steady 10 fps = real walking speed.
REM v6 remembers 16.8 s (v5: 4.8 s): turn away for a few seconds and turn back to test it.
REM For a smooth walk, pause training first (Ctrl+C in the training window; it saves),
REM then double-click its _resume.bat afterwards.
title Dream World - play
cd /d "%~dp0"
set CK=
if /i "%~1"=="v9" set CK=train\checkpoints\dream_v9\v9_pick.pt
if /i "%~1"=="v11" set CK=train\checkpoints\dream_v11\v11_pick.pt
REM play.bat fuel  -> LEARNED FUEL (2026-10-23 night): v9 + a real reference frame every 1 s (L toggles it)
if /i "%~1"=="fuel" set CK=train\_smoke\v9\checkpoints\fuel_s0\judge_fuel_s0.pt
if /i "%~1"=="v9night" set CK=train\checkpoints\dream_v9\best_melt.pt
if /i "%~1"=="v9preview" set CK=train\checkpoints\dream_v9_preview\v9A_sanity.pt
if /i "%~1"=="v8" set CK=train\checkpoints\dream_v8\final.pt
if /i "%~1"=="v8best" set CK=train\checkpoints\dream_v8\best.pt
if /i "%~1"=="v7" set CK=train\checkpoints\dream_v7\best.pt
if /i "%~1"=="v7steer" set CK=train\checkpoints\dream_v7\best_steer.pt
if /i "%~1"=="v6" set CK=train\checkpoints\dream_v6\best.pt
if /i "%~1"=="v5" set CK=train\checkpoints\dream_v5\best.pt
if /i "%~1"=="v4" set CK=train\checkpoints\dream_v4\step0073130.pt
if /i "%~1"=="v3" set CK=train\checkpoints\dream_v3\step0023524.pt
if /i "%~1"=="v2" set CK=train\checkpoints\dream_v2\best_step0025225.pt
if /i "%~1"=="v1" set CK=train\checkpoints\dream_v1\latest.pt
if /i "%~1"=="v0" set CK=train\checkpoints\dream_v0\latest.pt
if "%CK%"=="" if exist "train\checkpoints\dream_v9\v9_pick.pt" set CK=train\checkpoints\dream_v9\v9_pick.pt
if "%CK%"=="" if exist "train\checkpoints\dream_v8\final.pt" set CK=train\checkpoints\dream_v8\final.pt
if "%CK%"=="" if exist "train\checkpoints\dream_v7\best.pt" set CK=train\checkpoints\dream_v7\best.pt
if "%CK%"=="" if exist "train\checkpoints\dream_v6\best.pt" set CK=train\checkpoints\dream_v6\best.pt
if "%CK%"=="" if exist "train\checkpoints\dream_v5\best.pt" set CK=train\checkpoints\dream_v5\best.pt
if "%CK%"=="" set CK=train\checkpoints\dream_v3\step0023524.pt
echo Playing %CK%
data\venv\Scripts\python.exe train\play.py --ckpt %CK% %2 %3 %4
pause
