# Dream World: version history

A small diffusion model dreams the next frame of a street walk. You steer it with WASD.
"Melt" = the moment the dream stops looking like a place. Lower melt score = better.

## v0: first dream
- 39.3M-parameter EDM U-Net. 64x112 frames. 4 frames of memory.
- Data: 3 Walking Tours cities, 3.6 h, 129,773 frames, faces blurred.
- Keys came from optical flow (mostly W, ~62%; turns ~18%).
- First frame looked right. Then it melted into pink haze in 2-3 s.
- Fix with no training: 10 clean-up steps + a colour anchor. Melt 6.5 s -> never.

## v1: more memory
- 8 frames of memory. Practice on its own dreams (25% of each batch).
- Melt times jumped around from 3 to 30 s. No clear trend.
- Lesson: one test walk is too noisy to judge a model.

## v2: bigger frames
- 96x168 frames, warm-started from v1. Melt on 3 unseen walks: 5.5 / 10.6 / 10.1 s.

## v3: long memory, more cities
- 8 recent + 8 far memory frames (~4.8 s). A colour "fingerprint" input.
- Data: 9 cities + Wildlife. 14.4 h, 518k frames.
- New fair test: 4 walks x 2 seeds.
- Went wrong: every save turned into ripple texture by 3-5 s.
  The meters read ripples as "sharp", so they were fooled.
- Real peak: clean time 2.0 s (median).

## The hunt (tests between v3 and v4)
- Other sampler settings fooled the meters again. So a "scene meter" was added.
- Practising on 8 own frames (not 1-4) gave no gain.
- A 96.5M model learned faster per step than 39.6M, but not per minute.

## v4: big model from scratch
- 96.5M parameters, 119k steps, no crashes.
- Tied with v3 (clean time 3.7 s vs 4.0 s).

## The root cause (the big fix)
- Training noise never reached pure noise. A faint copy of the answer always showed through.
- But dreaming started from pure noise. So the first guess was bad, every time, since v0.
- Fix: start each frame from the previous frame + noise.
- First-frame error: 0.221 -> 0.037 (just copying the last frame scores 0.063).
- v3 clean walk: 4.0 s -> 11.3 s. Scene score 0.78 = real video.

## v5: the fix trained in
- Practice dreams also start from the previous frame.
- First save to stay clean for 20 of 20 s on all 10 test dreams.
- Fewer clean-up steps hurt: 10 steps = 20.0 s clean, 3 steps = 3.0 s. Keep 10.
- CUDA-graph replay: 6.2 -> 35.8 model frames per second.
- Went wrong: steering. Holding turn for 4 s turned the dream ~1 degree (real: ~90).

## v6: longer memory
- +8 "very far" memory frames (16.8 s total). 51.4M parameters.
- 40 s test walk. Clean ~25-28 s.

## v7: steering works
- Cause: the model copied the faint ghost of its start frame, so keys did nothing.
- Fix: "shifted ghost". Slide the previous frame by the turn before adding noise.
- Now it turns ~175 of 144 degrees asked. Turned views look like places.
- Bug: half-pixel slides blurred every frame. Fixed by sliding whole pixels only.
- Went wrong: turn back and it finds a NEW place (return score ~0.1-0.2).

## Panorama memory (no training)
- A 360-degree strip, painted with each dreamed frame at its heading.
- Turn back: 0.13 -> 0.88. Full circle: 0.02 -> 0.87.
- "Centre rule" (keep each column's most central view): 0.94-0.98.

## v8: practice with memory
- Trains on fake "patchwork" revisits (real footage almost never comes back).
- Low-noise version failed: every dream got softer.
- Old meters: 40 s walk clean mean 9.46 -> 12.09 s.
- Honest meters (below) later showed v8 slightly WORSE than v7.

## Honest meters
- The old meters flagged REAL video: mush 32%, curtains 44%, ripples 35%.
- New glitch meter: checks fine colour detail against the start frames.
- New melt critic: a 0.5M CNN, early vs late dream frames.
- Scores: real video 0.13, v7 0.57, v6 0.70, v8 0.69-0.72.

## v9: practise on wrecked memory
- 40% of rows get their 8 newest memory frames wrecked by the model itself.
  The answer stays the real next frame.
- Short test (3000 steps): melt 0.57 -> 0.44. Melt time 16 -> 24 s. Glitch 22 -> 39 s.
- Long run: the gain came in the first ~4k steps, then faded back to v7's level.
- More training itself erodes long walks: 3k steps 0.428, 6k 0.512, 9k 0.549.
- So the pick is the 3000-step model (0.438).

## Play upgrades (no training)
- Map memory: each ground spot keeps its own panorama. Out-and-back: 0.19 -> 0.84.
- Zoom walking: zoom the start frame 1.5% per W step. Melt 0.428, no glitch in 40 s.
  Walking feels faster: 35 -> 51 layout change at 4 s (real: 62).

## v10: experiment queue
- Many 3000-step sprints. Run-to-run noise ~0.05.
- Harder wrecks did best (0.402), but within noise. Longer was worse again.

## Dream fuel
- Blend a matching REAL frame into the whole ghost: failed. Washed-out mush (0.42-0.48).
- Edge fuel: fill only the strip a turn reveals, from a real frame.
  Spin melt 0.422 -> 0.221. Streaks 0.695 -> 1.012 (real ~1). 40 s walks 0.428 -> 0.397.

## Sharper eyes (2x upscaler, display only)
- 3.66M parameters. Detail 0.90 of real (plain stretch: 0.32). 1.5 ms per frame.
- No faces invented. Late mush becomes crisp mush.

## v11: dream at 2x (192x336)
- 94 GB face-blurred cache. GPU face blur: ~15 -> ~160 frames per second.
- Went wrong: melts sooner than v9 (0.596 vs 0.428).
- v11.1: 2x training noise helped (0.569 vs 0.639). A deeper middle did little (0.602).
- Long run stayed flat at 0.59-0.67. v9 stays the champion.

## Learned fuel
- A real reference frame as an extra memory input.
- At 3k steps: worse (0.491 vs 0.428). At 6k: better (0.439 vs 0.512).
- It improves with longer training, while plain v9 erodes.

## Painter v1
- 23.0M diffusion upscaler that remembers its last painted frame.
- Steadier (flicker 0.71 vs 0.92) but softer (detail 0.61 vs 0.88).
- Not an upgrade over sharper eyes yet.
