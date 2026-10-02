# FailureBench: frozen experimental protocol

Created 2 October 2026, before inspecting benchmark results. This is an experimental protocol, not a manuscript or an external preregistration. Its exact bytes and the source files are hashed in each run manifest.

## Question and candidate gap

How large is the difference between coverage for each fixed feature-loss pattern and coverage after an input-dependent pattern selection? What interval-width cost buys simultaneous protection, and can independently fitted mask-specific scales reduce that cost?

The candidate gap is an empirical evaluation gap at the intersection of missing-feature conformal calibration and adaptive adversarial robustness. The max-score construction, normalization, and rank proof are established ideas. No claim of a first method, proved literature-wide novelty, or publishability is made. Generic missingness-aware uncertainty is already extensively studied. The strongest plausible contribution is a reproducible audit distinguishing endpoints that are easily conflated, with an explicit counterexample and failure boundaries.

## Data and splits

Five public regression tasks: diabetes, concrete strength, energy heating load, airfoil sound pressure, and red-wine quality. Two synthetic tasks: heteroskedastic linear and nonlinear regression with correlated covariates. Wine quality is a discrete numeric target; ties are retained. Every dataset gets ten prespecified random splits. The data split is 50% model training, 20% scale fitting, 15% conformal calibration, 15% testing. Splits are disjoint at original-row level, before any augmentation. For public datasets, exact duplicate predictor rows are reduced to one deterministic representative before splitting; the removed counts are logged. This avoids identical feature rows crossing splits, but does not establish independence of physical specimens.

Imputation medians, feature scaling, response scaling, feature rankings, and predictor fits use model-training rows only. Mask scales use scale-fit rows only. Conformal thresholds use calibration rows only. Test outcomes are accessed only by evaluation and the explicitly labelled oracle diagnostic. No test-based hyperparameter selection or favorable seed selection. Public datasets are convenience benchmarks, not evidence about deployed sensors or clinical systems.

## Predictors and feature loss

Fixed models: standardized ridge (alpha=1) and histogram gradient boosting (120 iterations, 15 leaf nodes, learning rate .08, L2=1, minimum leaf size 20, early stopping disabled). Two training arms: complete inputs and complete inputs plus one independently sampled masked copy per training row. Missingness indicators are appended in both arms; unavailable raw features are imputed to their training medians before applying training-fitted standardization.

Features are ranked by absolute training correlation with the response, breaking ties by column index. Finite mask families contain the clean input and single-feature losses for the top 2, top 4, or all features. Every candidate is enumerated exactly. There is no certificate for arbitrary feature subsets. The augmented training arm samples from the all-feature single-loss family, independently of calibration and test data.

## Methods

1. Clean CP: calibrate on complete inputs.
2. Random-mask CP: one uniform random family mask per original calibration row; targets a matching random-mask distribution.
3. Pooled augmentation: flatten all row-mask residuals, a deliberately naive comparator without the usual rank guarantee for dependent replicas.
4. Mask-matched CP: a separate corrected quantile for each fixed mask; the simple exact-mask augmentation special case for initially complete source data. This is not a full reproduction of Zaffran et al.'s CQR/MDA-Nested algorithms.
5. Mask envelope: use the largest mask-matched threshold for every mask. Fixed-mask validity does not imply simultaneous validity.
6. Bonferroni CP: calibrate each mask at alpha/K. Valid simultaneous baseline; infinite widths are retained when calibration size is insufficient.
7. Row-max CP: calibrate one maximum residual per original row over all family masks. Standard split-conformal rank protection for simultaneous coverage.
8. Scaled row-max CP: divide each mask residual by a positive scale fitted on the independent scale-fit split, then calibrate each row maximum. Scale = 75th percentile of scale-fit absolute residuals with a .05 floor in training-response-standard-deviation units. No test-dependent scale fitting.
9. Policy-matched CP: calibrate residuals for a fixed, label-free prediction-shift selection policy. Valid for exchangeable source rows and that frozen policy; no arbitrary-policy or mask-conditional certificate.

All quantiles use rank ceil((n+1)(1-alpha)). When this rank exceeds n, the threshold is positive infinity. Interpolated quantiles are never used for conformal thresholds. Normalization quantiles on the scale-fit split may use interpolation because they do not supply the conformal guarantee.

## Endpoints and policies

Primary endpoint: simultaneous in-family coverage, the fraction of test rows for which every family mask's interval contains the same true response. Also report mean width across those masks. These are stronger targets than worst fixed-mask coverage (the minimum of mask-wise empirical coverage), which is separately reported.

Label-free policies: uniform random mask; saturation (choose the largest absolute standardized value among the droppable features); prediction shift (largest absolute masked prediction deviation from the complete-input prediction). They are specified before results. Saturation and prediction shift assume access to the complete input at the stage that induces the loss; the deployed predictor receives only the masked input. They are stress-test generators, not failure detectors deployable after the values have disappeared.

Oracle diagnostic: select the mask with greatest residual/interval-radius ratio for each method, using the true outcome. This is an exhaustive worst-case stress test, never a practical predictor. Its coverage equals simultaneous coverage. Its width is policy dependent; the primary simultaneous width is the family-average width.

Outside-family diagnostic: use prediction shift to choose among two-feature losses of the top four features. Transfer each method's threshold conservatively to a pair (maximum singleton threshold; maximum singleton scale for the scaled method). This extension is explicitly heuristic and uncertified. Out-of-family failures must remain in the results.

Secondary metrics: normalized and original-unit width, selected prediction RMSE, worst fixed-mask coverage, clean-mask coverage, and Wilson intervals for test coverage. Wilson intervals describe one finite test sample under an independent-Bernoulli approximation, not the conformal theorem. Infinite-width fraction and calibration cost are logged.

## Comparisons, uncertainty, and reporting

All methods share predictor, splits, candidate predictions, and random-mask draws inside a dataset/model/seed/training cell. Primary comparisons use the all-feature family and alpha=.1. Ablations cover family sizes, alpha=.05, both models, both training arms, and scaled versus unscaled row maxima.

Report all dataset/model/arm combinations, not only the strongest case. Summaries give mean across the ten split seeds and paired seed-bootstrap 95% intervals for differences. Repeated splits overlap within a fixed dataset; these intervals measure split stability conditional on that benchmark, not independent-population evidence. Datasets are not treated as independent replications of a common population. Cross-dataset aggregates are descriptive equal-weight averages, with no significance claim.

Results may support an audit/reproducibility paper, or may show that the candidate is too incremental. A successful result requires a visible endpoint gap and transparent efficiency tradeoffs. A failure to improve efficiency is still recorded. Manuscript drafting is deferred at the user's request.
