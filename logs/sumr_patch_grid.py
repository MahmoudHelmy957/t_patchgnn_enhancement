#!/usr/bin/env python3
import pandas as pd

RAW_CSV = "physio_patch_grid_raw.csv"
OUT_CSV = "physio_patch_grid_summary.csv"

def main():
    df = pd.read_csv(RAW_CSV)

    # Ensure numeric types (sometimes CSVs load as strings)
    for col in ["MSE", "RMSE", "MAE", "MAPE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group summary with mean/std
    summary = (
        df.groupby(["dataset", "patch_size", "stride"])
          .agg({
              "MSE":  ["mean", "std"],
              "RMSE": ["mean", "std"],
              "MAE":  ["mean", "std"],
          })
          .reset_index()
    )

    # Flatten MultiIndex columns: ("MSE","mean") -> "MSE_mean"
    summary.columns = ["_".join(tup).strip("_") for tup in summary.columns.to_flat_index()]

    # (Optional) nicer ordering of columns
    ordered_cols = [
        "dataset", "patch_size", "stride",
        "MSE_mean", "MSE_std",
        "RMSE_mean", "RMSE_std",
        "MAE_mean", "MAE_std",
    ]
    # keep any unexpected extras at the end
    summary = summary[[c for c in ordered_cols if c in summary.columns] +
                      [c for c in summary.columns if c not in ordered_cols]]

    # Save
    summary.to_csv(OUT_CSV, index=False)

    # Pretty print
    with pd.option_context("display.float_format", "{:0.6f}".format):
        print("\n=== Summary ===")
        print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
