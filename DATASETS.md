# Data attribution and caveats

Original archives are cached in data/; per-dataset provenance JSON files record download URLs, SHA-256 hashes, row counts, feature names, and removed exact-duplicate predictor rows. No dataset is silently replaced when a download fails.

| Dataset | Source and credit | Target | Original size | Use and limits |
| --- | --- | --- | --- | --- |
| Diabetes | [scikit-learn packaged dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset), from Efron et al., Least Angle Regression (2004) | Numeric progression measure | 442 rows / 10 predictors | Convenience regression task. No clinical efficacy or deployment claim. See packaged DESCR and scikit-learn distribution attribution. |
| Concrete | [UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), I-Cheng Yeh (1998), DOI 10.24432/C5PK67 | Strength in MPa | 1030 / 8 | Artificial feature deletion; no real telemetry failure data. UCI CC BY 4.0. |
| Energy | [UCI](https://archive.ics.uci.edu/dataset/242/energy+efficiency), Athanasios Tsanas and Angeliki Xifara (2012), DOI 10.24432/C51307 | Heating load | 768 / 8 | Building energy values are simulated; this is a public benchmark, not measured physical sensor reliability. Cooling load is excluded from inputs. UCI CC BY 4.0. |
| Airfoil | [UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), NASA, Roberto Lopez (2014), DOI 10.24432/C5VW2C | Sound pressure | 1503 / 5 | Source rows may share underlying experimental settings; random row splits do not prove physical-specimen independence. UCI CC BY 4.0. |
| Red wine | [UCI](https://archive.ics.uci.edu/dataset/186/wine+quality), Paulo Cortez, Antonio Cerdeira, Fernando Almeida, Telmo Matos, Jose Reis (2009), DOI 10.24432/C56S3T | Ordinal quality as numeric regression | 1599 / 11 | Discrete targets/ties retained. Exact duplicate predictor rows reduced before splitting. No taste or production deployment claim. UCI CC BY 4.0. |
| Synthetic linear | Local frozen generator seed 19283 | Correlated linear function plus heteroskedastic Gaussian noise | 5000 / 10 | Controlled i.i.d. simulation; useful for checking assumptions and robustness. |
| Synthetic nonlinear | Local frozen generator seed 92387 | Sinusoid, interaction, quadratic term plus heteroskedastic Gaussian noise | 5000 / 10 | Controlled i.i.d. simulation; useful for model misspecification. |

Exact duplicate predictor rows are identified without outcomes; the first original source row is retained deterministically. This avoids identical input rows in different splits, but changes the weighting of the original benchmark distribution and discards repeated measurements. All resulting counts are reported. The finite-sample theorem applies to exchangeable original source rows under its assumptions; real dataset provenance does not establish those assumptions automatically.
