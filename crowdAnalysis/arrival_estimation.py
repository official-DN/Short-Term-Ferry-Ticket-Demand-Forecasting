
"""
Estimate future passenger arrivals at a ferry terminal
from predicted ticket demand.

Modes:
1. Historical delay distribution
2. Configurable assumption-based estimation

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ArrivalModelConfig:
    interval_minutes: int = 30

    # Used only when historical arrival data is unavailable.
    # This is an assumption, not a learned value.
    fallback_arrival_rate: float = 0.80

    # Historical delay bucket limits.
    max_delay_minutes: int = 240

    # Minimum historical records required to learn
    # a reliable delay distribution.
    min_training_records: int = 30


class PassengerArrivalEstimator:
    """
    Estimate passenger arrivals from predicted ticket sales.

    Historical mode:
        Learns the time delay between ticket purchase
        and actual passenger arrival.

    Fallback mode:
        Applies a configured arrival proportion.
    """

    def __init__(
        self,
        config: Optional[ArrivalModelConfig] = None
    ):
        self.config = config or ArrivalModelConfig()

        self.delay_distribution = None
        self.is_fitted = False
        self.training_records = 0


    # Learn historical arrival delay distribution
    def fit(
        self,
        historical_arrivals: pd.DataFrame,
        purchase_col: str = "ticket_purchase_time",
        arrival_col: str = "passenger_arrival_time"
    ):
        """
        Learn passenger arrival delays from historical data.

        Required columns:
            ticket_purchase_time
            passenger_arrival_time

        Each row should represent one passenger/ticket
        with a known purchase and terminal arrival time.
        """

        df = historical_arrivals.copy()

        df[purchase_col] = pd.to_datetime(
            df[purchase_col],
            errors="coerce"
        )

        df[arrival_col] = pd.to_datetime(
            df[arrival_col],
            errors="coerce"
        )

        df = df.dropna(
            subset=[purchase_col, arrival_col]
        )

        # Calculate delay in minutes.
        df["delay_minutes"] = (
            df[arrival_col] - df[purchase_col]
        ).dt.total_seconds() / 60

        # Remove invalid records:
        # arrival before purchase or extreme delays.
        df = df[
            (df["delay_minutes"] >= 0)
            &
            (
                df["delay_minutes"]
                <= self.config.max_delay_minutes
            )
        ]

        self.training_records = len(df)

        if (
            self.training_records
            < self.config.min_training_records
        ):
            raise ValueError(
                "Insufficient historical arrival records. "
                "Collect more paired purchase/arrival records "
                "or use fallback estimation."
            )

        interval = self.config.interval_minutes

        # Assign delays into interval-sized buckets.
        df["delay_bucket"] = (
            np.floor(
                df["delay_minutes"] / interval
            ).astype(int)
        )

        # Calculate probability of each delay bucket.
        probabilities = (
            df["delay_bucket"]
            .value_counts(normalize=True)
            .sort_index()
        )

        # Include missing buckets with probability zero.
        max_bucket = int(
            self.config.max_delay_minutes // interval
        )

        probabilities = probabilities.reindex(
            range(max_bucket + 1),
            fill_value=0.0
        )

        # Normalize probabilities.
        probabilities = (
            probabilities / probabilities.sum()
        )

        self.delay_distribution = probabilities

        self.is_fitted = True

        return self

    
    # Predict future passenger arrivals
    def predict_arrivals(
        self,
        demand_forecast: pd.DataFrame,
        demand_col: str = "predicted_sales",
        time_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Estimate future arrivals from predicted ticket demand.

        demand_forecast must contain:
            predicted_sales

        The index or time_col should represent the START
        of each ticket-sales interval.

        Returns:
            DataFrame with:
                predicted_sales
                estimated_arrivals
        """

        df = demand_forecast.copy()

        if time_col is not None:
            df[time_col] = pd.to_datetime(
                df[time_col],
                errors="coerce"
            )

            df = df.set_index(time_col)

        if demand_col not in df.columns:
            raise ValueError(
                f"Missing forecast column: {demand_col}"
            )

        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        df = df.loc[
            ~df.index.isna()
        ].sort_index()

        if df.empty:
            raise ValueError(
                "No valid forecast intervals found."
            )

        # Forecast ticket demand should be non-negative.
        demand = (
            df[demand_col]
            .fillna(0)
            .clip(lower=0)
            .astype(float)
        )

        # Initialize arrival predictions.
        arrivals = pd.Series(
            0.0,
            index=df.index
        )


        # Historical learned delay model
        if self.is_fitted:

            interval = self.config.interval_minutes

            for delay_bucket, probability in (
                self.delay_distribution.items()
            ):

                if probability <= 0:
                    continue

                # Shift future arrivals by the
                # learned purchase-to-arrival delay.
                shift = (
                    int(delay_bucket)
                    * interval
                )

                shifted_index = (
                    demand.index
                    + pd.Timedelta(
                        minutes=shift
                    )
                )

                shifted_demand = pd.Series(
                    demand.values * probability,
                    index=shifted_index
                )

                # Add contributions from all purchase
                # cohorts to their predicted arrival
                # intervals.
                arrivals = arrivals.add(
                    shifted_demand,
                    fill_value=0
                )


        # Assumption-based fallback model

        else:

            rate = (
                self.config.fallback_arrival_rate
            )

            arrivals = demand * rate

        # Return forecast and arrival estimates.
        result = pd.DataFrame(
            {
                "predicted_sales": demand,
                "estimated_arrivals": arrivals
            }
        )

        result["estimated_arrivals"] = (
            result["estimated_arrivals"]
            .clip(lower=0)
        )

        return result


    # Calculate cumulative estimated arrivals

    def predict_cumulative_arrivals(
        self,
        demand_forecast: pd.DataFrame,
        demand_col: str = "predicted_sales"
    ) -> pd.DataFrame:
        """
        Calculate cumulative estimated arrivals.

        Useful for analyzing future passenger arrival
        pressure across multiple intervals.
        """

        result = self.predict_arrivals(
            demand_forecast,
            demand_col=demand_col
        )

        result["cumulative_arrivals"] = (
            result["estimated_arrivals"]
            .cumsum()
        )

        return result
