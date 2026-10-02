"""Original public-source downloads, deterministic deduplication, provenance."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from sklearn.datasets import load_diabetes


UCI = {
    "concrete": (165, "concrete+compressive+strength", "10.24432/C5PK67"),
    "energy": (242, "energy+efficiency", "10.24432/C51307"),
    "airfoil": (291, "airfoil+self+noise", "10.24432/C5VW2C"),
    "wine_red": (186, "wine+quality", "10.24432/C56S3T"),
}


def synthetic(name: str, n: int = 5000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Dataset-generation randomness is fixed independently of experiment split seeds.
    rng = np.random.default_rng(19283 if name == "synthetic_linear" else 92387)
    latent = rng.normal(size=(n, 1))
    x = .4 * latent + np.sqrt(.84) * rng.normal(size=(n, 10))
    if name == "synthetic_linear":
        mean = 3 * x[:, 0] + 2 * x[:, 1] - 1.5 * x[:, 2] + .5 * x[:, 3]
        noise = (.3 + .5 * np.abs(x[:, 4])) * rng.normal(size=n)
    elif name == "synthetic_nonlinear":
        mean = 2 * np.sin(x[:, 0]) + 1.5 * x[:, 1] * x[:, 2] + x[:, 3] ** 2 - x[:, 4]
        noise = (.25 + .5 * (x[:, 5] > 0)) * rng.normal(size=n)
    else:
        raise ValueError(name)
    return x, mean + noise, [f"x{j}" for j in range(x.shape[1])]


def load_dataset(name: str, cache: Path, synthetic_n: int = 5000):
    cache.mkdir(parents=True, exist_ok=True)
    source: dict = {"name": name}
    if name.startswith("synthetic_"):
        x, y, features = synthetic(name, synthetic_n)
        source.update(source="local controlled generator", license="project MIT license")
    elif name == "diabetes":
        ds = load_diabetes(scaled=False)
        x, y, features = ds.data.astype(float), ds.target.astype(float), list(ds.feature_names)
        source.update(source="scikit-learn packaged diabetes (Efron et al.)",
                      url="https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
                      license="see scikit-learn source and dataset attribution")
        (cache / "diabetes_description.txt").write_text(ds.DESCR, encoding="utf-8")
    else:
        dataset_id, slug, doi = UCI[name]
        url = f"https://archive.ics.uci.edu/static/public/{dataset_id}/{slug}.zip"
        raw_path = cache / f"{name}.zip"
        if not raw_path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "FailureBench-research/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_path.write_bytes(response.read())
        raw = raw_path.read_bytes()
        source.update(source="UCI Machine Learning Repository", url=url,
                      doi=doi, license="CC BY 4.0", raw_sha256=hashlib.sha256(raw).hexdigest())
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if name == "concrete":
                import xlrd
                member = next(p for p in archive.namelist() if p.lower().endswith(".xls"))
                sheet = xlrd.open_workbook(file_contents=archive.read(member)).sheet_by_index(0)
                features = [str(sheet.cell_value(0, j)) for j in range(sheet.ncols - 1)]
                values = np.array([sheet.row_values(i) for i in range(1, sheet.nrows)], dtype=float)
                x, y = values[:, :-1], values[:, -1]
            elif name == "energy":
                import openpyxl
                member = next(p for p in archive.namelist() if p.lower().endswith(".xlsx"))
                workbook = openpyxl.load_workbook(io.BytesIO(archive.read(member)), read_only=True, data_only=True)
                sheet = workbook.active
                rows = list(sheet.values)
                features = [str(v) for v in rows[0][:8]]
                values = np.array([r[:10] for r in rows[1:] if r[0] is not None], dtype=float)
                x, y = values[:, :8], values[:, 8]  # heating load; cooling load is not an input
                workbook.close()
            elif name == "airfoil":
                member = next(p for p in archive.namelist() if p.lower().endswith(".dat"))
                values = np.loadtxt(io.BytesIO(archive.read(member)))
                x, y = values[:, :-1], values[:, -1]
                features = ["frequency", "angle", "chord", "velocity", "displacement_thickness"]
            else:
                member = next(p for p in archive.namelist() if p.lower().endswith("winequality-red.csv"))
                raw_csv = archive.read(member).decode("utf-8-sig")
                features = raw_csv.splitlines()[0].replace('"', "").split(";")[:-1]
                values = np.loadtxt(io.StringIO(raw_csv), delimiter=";", skiprows=1)
                x, y = values[:, :-1], values[:, -1]
            source["archive_member"] = member
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: unexpected missing/nonfinite source data")
    original_n = len(y)
    _, unique_ids = np.unique(x, axis=0, return_index=True)
    unique_ids.sort()
    x, y = x[unique_ids], y[unique_ids]
    source.update(original_rows=original_n, rows=len(y), features=features,
                  removed_duplicate_predictor_rows=original_n - len(y),
                  data_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
    (cache / f"{name}_provenance.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    return x, y, features, source
