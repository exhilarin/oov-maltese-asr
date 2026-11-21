### Baselines (FLEURS mt_mt test)

| model | revision | WER % | CER % | RTF | hardware | utts |
|---|---|---|---|---|---|---|
| nvidia/canary-1b-v2 | d455706339 | 22.15 | 7.10 | 0.041 | Tesla T4 | 926 |
| nvidia/parakeet-tdt-0.6b-v3 | 541d1f99c6 | 24.77 | 6.45 | 0.0101 | Tesla T4 | 926 |
| carlosdanielhernandezmena/wav2vec2-large-xlsr-53-maltese-64h | 0468585cd5 | 25.41 | 7.38 | 0.0072 | Tesla T4 | 926 |
| openai/whisper-small | 973afd2496 | 111.93 | 54.01 | 0.1934 | Tesla T4 | 926 |

### Ours (FLEURS mt_mt test)

| model | revision | WER % | CER % | RTF | hardware | utts |
|---|---|---|---|---|---|---|
| whisper-small-mt | local | 34.24 | 9.35 | 0.0122 | Tesla T4 | 926 |
