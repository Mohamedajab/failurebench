# Executed verification

* Research-integrity tests: **20 passed**, covering finite-sample ranks, infinity at insufficient calibration sizes, ties, original-row split disjointness, frozen training preprocessing, scale positivity, label-free policy interfaces, the fixed-mask counterexample, and a 4,000-trial exchangeable-rank simulation.
* Small end-to-end experiment: eight predictor fits on diabetes and a 1,000-row controlled simulation, completed before the full benchmark.
* Full benchmark: **280 predictor fits**, **90,720 metric cells**, five public and two synthetic tasks; completed successfully in **57.2 seconds** on CPU.
* Every metric cell checked for completeness, unique keys, original-row coverage counts, Wilson interval consistency, width bounds, and agreement of exhaustive oracle coverage with simultaneous coverage.
* Every saved prediction file independently audited: **280 files** and **3,360 robust endpoint cells** recomputed using sorted-list ranks rather than the runner's calibration function. Counts and widths match; all original-row partitions are complete and disjoint.
* Raw CSV hashes verified against the completed-run manifest. Prediction hashes are in independent_audit.json. The current analysis/audit source hashes are recorded separately because reporting code was refined after raw benchmark execution; the predictor/calibration/runner code used for the raw run is unchanged.
* The three exported figures were visually inspected. Results and uncertainty caveats are in results/full/RESULTS.md.

No manuscript, external preregistration, uploaded preprint, remote repository, or publication was created.
