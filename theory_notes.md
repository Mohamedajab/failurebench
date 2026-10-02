# Mathematical notes, not a manuscript

## Distinct evaluation targets

For a frozen predictor f and finite mask family M, let r_m(X,Y) = |Y-f(T_m(X))|. Initially complete feature vectors (X,Y) are the exchangeable units. T_m removes features and applies training-fitted imputation. Let C_m denote an interval at mask m.

* Fixed-mask coverage: for each fixed m, P(Y in C_m(T_m(X))) >= 1-alpha. This is an unconditional probability for a counterfactual fixed mask, not coverage conditional on a realized input-dependent mask assignment.
* Simultaneous coverage: P(for every m in M, Y in C_m(T_m(X))) >= 1-alpha.
* Selected-policy coverage: P(Y in C_A(X)(T_A(X)(X))) >= 1-alpha, for a specified selector A. To avoid ambiguous notation, the selected interval is simply the member indexed by A(X).

Simultaneous coverage implies selected-policy coverage for every in-family selector, even one that depends on X, Y, or the calibration thresholds. Fixed-mask coverage alone does not. This does not establish conditional coverage given a selected mask, nor conditional coverage at every feature vector.

## Counterexample

Let U be uniform on [0,1] and Y=0. At radius zero, suppose mask A predicts 1 only on [0,.1), while mask B predicts 1 only on [.1,.2). Both otherwise predict 0. Each fixed-mask interval covers exactly 90% of outcomes. A selector that chooses A on the first region and B on the second covers only 80%. The selector is a function of U and needs no outcome label. Simultaneous coverage is also 80%. The construction shows a logical distinction, not a claim about typical real failure mechanisms or a new theorem.

## Normalized row-max construction

Fit f, the mask family, imputation, and positive mask scales s_m on data independent of calibration. Define one score per ORIGINAL calibration row:

R_i = max over m in M of (r_m(X_i,Y_i) / s_m).

Set k = ceil((n_cal+1)(1-alpha)) and q = the kth smallest R_i; use infinity when k>n_cal. Return C_m(T_m(x)) = [f(T_m(x))-q*s_m, f(T_m(x))+q*s_m]. The unit-scale case is row-max CP.

Conditional on the independent fitted objects, calibration scores and the future row score are exchangeable. The usual corrected-rank argument gives P(R_test <= q) >= 1-alpha. The event R_test <= q is exactly simultaneous membership of Y_test in every C_m. No union bound over masks is required. Ties make the non-randomized construction conservative. The probability averages over calibration and the future original row; it is not a guarantee conditional on a realized calibration set. This is standard split-conformal theory applied to a finite maximum score, not a claimed novel result.

The proof remains valid for training-selected mask families and scales when they are independent of calibration. It does not survive unrestricted adaptation of those objects to calibration labels. It also does not certify unenumerated two-feature losses, general covariate/label shift, dependence between physical specimens, or arbitrary natural missingness mechanisms with no complete calibration data.

## Established comparators and costs

Bonferroni calibrates each mask at alpha/|M|, so a union bound gives simultaneous protection. With n calibration rows it may require infinite thresholds when ceil((n+1)(1-alpha/|M|))>n. Row-max shares the same n original calibration units and costs O(n|M|) residual evaluations. For d singleton failures |M|=d+1, so enumeration is linear in d. Enumerating all feature subsets would be exponential and is outside this project.

Policy-matched CP applies ordinary split conformal to |Y-f(T_A(X)(X))| for a fixed label-free selector. It can be narrower, but transferring its threshold to another selector has no general guarantee. All counterfactual feature values may be used at the loss-generation stage; deployed interval inference uses only the observed mask and imputed masked input.

Sources: [Zaffran et al.](https://proceedings.mlr.press/v202/zaffran23a.html), [Ghosh et al.](https://arxiv.org/abs/2307.16360), and [Massena et al.](https://arxiv.org/abs/2506.05434). Broader novelty and clinical transfer remain unverified.
