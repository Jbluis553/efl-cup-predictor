#!/usr/bin/env python3
"""Muestra las predicciones del día desde la carpeta de salida `predictions`.

Ejemplo:
    python daily_predictions.py
    python daily_predictions.py --date 2026-11-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "kickoff",
    "home_team",
    "away_team",
    "P(home)",
    "P(draw)",
    "P(away)",
    "P(over_2_5)",
    "P(btts)",
]


def find_predictions_file(date: str | None = None, predictions_dir: Path | str = "predictions") -> Path:
    """Devuelve el CSV de predicciones más apropiado para la fecha indicada."""
    predictions_dir = Path(predictions_dir)
    if not predictions_dir.exists():
        raise FileNotFoundError(
            "No existe la carpeta 'predictions'. Ejecuta primero: python -m src.predict"
        )

    files = sorted(predictions_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No hay archivos de predicciones en 'predictions/'. Ejecuta primero: python -m src.predict"
        )

    if date:
        target = date
        matches = [f for f in files if f.stem.endswith(target)]
        if matches:
            return matches[-1]
        raise FileNotFoundError(f"No existe un fichero de predicciones para la fecha {date}.")

    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    today_matches = [f for f in files if f.stem.endswith(today)]
    if today_matches:
        return today_matches[-1]

    return files[-1]


def print_predictions(file_path: Path, limit: int | None = None) -> None:
    """Imprime la tabla de predicciones en consola."""
    df = pd.read_csv(file_path)
    if df.empty:
        print(f"No hay predicciones para {file_path.name}.")
        return

    if limit is not None:
        df = df.head(limit)

    columns = [c for c in DEFAULT_COLUMNS if c in df.columns]
    if not columns:
        columns = list(df.columns[:8])

    display = df[columns].copy()
    if "kickoff" in display.columns:
        display["kickoff"] = pd.to_datetime(display["kickoff"], errors="coerce")
        display["kickoff"] = display["kickoff"].dt.strftime("%Y-%m-%d %H:%M UTC")

    print(f"\nPredicciones: {file_path.name}\n")
    print(display.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestra las predicciones del día.")
    parser.add_argument(
        "--date",
        type=str,
        help="Fecha en formato YYYY-MM-DD. Si no se indica, usa la de hoy.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Número máximo de partidos a mostrar.",
    )
    args = parser.parse_args()

    try:
        file_path = find_predictions_file(args.date)
        print_predictions(file_path, limit=args.limit)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
