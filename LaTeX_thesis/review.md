# Thesis Review — Remaining Open Items

All priority fixes from the original review have been applied (template compliance, DL hyperparameters, CSP n_components documentation, EEGNet/ShallowConvNet parameter counts, preprocessing division of labour, TL baseline disclaimer, Chapter 6/7 expansions, Wilcoxon statistical tests, BibTeX type fixes, `material.tex` deleted). The items below are what still needs author attention.

## Minor Content Items

1. **Ch. 2, line ~194 — ShallowConvNet defaults**: "A bank of 40 filters of length 25 samples is convolved along the time axis." This assumes braindecode's `ShallowFBCSPNet` defaults at the paper's original 250 Hz sampling rate. At this thesis's 500 Hz, a kernel of 25 samples covers only 50 ms. Verify that `methods/shallow.py` is actually instantiating with `n_filters_time=40, filter_time_length=25` (neither is set explicitly in `shallow.py`, so the braindecode defaults apply — confirm which they are and state them explicitly).

2. **Ch. 2, line ~288 — 18 EEG channels claim**: "The raw recordings contain 18 EEG channels sampled at either 500 Hz (Mochura protocol) or 1000 Hz (Saleh protocol)." Not verified against the actual data files. Load one `.vhdr` per protocol and confirm the channel count and sampling rate.

3. **Ch. 5 — mixed granularity between Table 5.1 and Table 5.2**: Table 5.1 aggregates over 29 subjects; Table 5.2 over 18 session-entries (9 subjects × 2 sessions — see line 648). The two means are not on equal statistical footing. Add a one-sentence caveat under Table 5.2 explicitly noting this difference.

4. **Ch. 6, line ~796 — "full 64-channel recordings" claim**: EEGNet was validated by Lawhern et al. on 4-, 64-, and 256-channel setups, not exclusively 64-channel. Consider rewording to "typically larger channel counts (e.g. 22 or 64)".

5. **Bibliography — `klem1999ten`**: No `number` field and only `pages={3--6}`. Verify the supplement/volume number against the source.

## Optional Additions

6. **Extra references**: the modern MOABB paper (Aristimunha et al., 2023) could be cited alongside the existing `jayaram2018moabb`. Skipped because `jayaram2018moabb` is sufficient for the thesis's claims.

7. **Double bandpass application**: the 8–30 Hz filter is applied both inside `custom_dataset._load_and_prepare_raw` and by MOABB's `MotorImagery(fmin=8, fmax=30)` in `run_pipeline.py`. Functionally harmless, but worth documenting or removing one of them for cleanliness.

## Author-side Verification

8. **Compile**: no `pdflatex` available in the current WSL environment. Please build locally and check:
    - `tab:wilcoxon` (Ch. 6) floats to a sensible location.
    - `\listoffigures`, `\listoftables`, `\backpage` render correctly in the back matter.
    - `\frontpages[notm]` produces the intended front matter (no trademark declaration).
    - All cross-references resolve (`sec:methods_csp`, `tab:results_custom`, `tab:results_tl`).
