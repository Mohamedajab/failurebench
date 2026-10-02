"""Exact finite-grid illustration: python -m failurebench.counterexample."""
import json
import numpy as np


def main():
    u = (np.arange(10000) + .5) / 10000
    residuals = np.stack((u < .1, (u >= .1) & (u < .2)), axis=1)
    choices = np.where(u < .1, 0, 1)  # depends only on U; no outcome access
    output = {
        "fixed_mask_A_coverage": float((~residuals[:, 0]).mean()),
        "fixed_mask_B_coverage": float((~residuals[:, 1]).mean()),
        "label_free_selected_coverage": float((~residuals[np.arange(len(u)), choices]).mean()),
        "simultaneous_coverage": float((~residuals.any(axis=1)).mean()),
        "interpretation": "Population/grid counterexample; this is not a finite-sample calibration experiment.",
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
