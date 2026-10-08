"""Simple linear regression: predict salary from years of experience."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "Salary_dataset.csv"
DEFAULT_OUTPUT = ROOT / "images"

FEATURE = "YearsExperience"
TARGET = "Salary"


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset and drop the unnamed index column if present."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    missing = {FEATURE, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")
    return df.dropna()


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Return common regression metrics."""
    return {
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_results(df, model, x_test, y_test, save_path: Path, show: bool) -> None:
    """Plot all data, the held-out test points and the fitted line."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[FEATURE], df[TARGET], color="lightgray", label="All data")
    ax.scatter(x_test, y_test, color="tab:blue", label="Test set")

    x_line = pd.DataFrame({FEATURE: [df[FEATURE].min(), df[FEATURE].max()]})
    ax.plot(x_line[FEATURE], model.predict(x_line), color="tab:red", label="Regression line")

    ax.set_xlabel("Years of experience")
    ax.set_ylabel("Salary")
    ax.set_title("Salary vs. years of experience")
    ax.legend()
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="path to the CSV file")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="where to save the plot")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="display the plot window")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows\n")
    print(df.describe().round(2), "\n")

    x_train, x_test, y_train, y_test = train_test_split(
        df[[FEATURE]], df[TARGET], test_size=args.test_size, random_state=args.seed
    )

    model = LinearRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Model: Salary = {model.intercept_:,.0f} + {model.coef_[0]:,.0f} * YearsExperience\n")

    comparison = pd.DataFrame(
        {"YearsExperience": x_test[FEATURE], "Actual": y_test, "Predicted": y_pred.round(0)}
    ).sort_index()
    print(comparison.to_string(index=False), "\n")

    for name, value in evaluate(y_test, y_pred).items():
        print(f"{name}: {value:,.4f}")

    plot_path = args.output_dir / "regression_plot.png"
    plot_results(df, model, x_test, y_test, plot_path, args.show)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
