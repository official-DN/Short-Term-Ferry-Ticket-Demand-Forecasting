
"""
Predict future passenger queue at a ferry terminal.

Queue equation:
    Q_t = Q_(t-1) + A_t - B_t

Actual boarding:
    B_t = min(Q_(t-1) + A_t, C_t)

Where:
    Q_t = queue at the end of interval
    A_t = passenger arrivals during interval
    B_t = passengers boarded during interval
    C_t = available boarding capacity during interval

"""

from typing import Optional, Union

import numpy as np
import pandas as pd


class QueuePredictionModel:
    """
    Predict future ferry terminal queue size.

    Inputs:
        estimated_arrivals:
            Passenger arrivals per interval.

        boarding_capacity:
            Maximum passengers that can board during
            each interval.

        initial_queue:
            Queue at the beginning of the forecast.

    Optional:
        actual_boarded:
            Actual or independently predicted boarding
            counts per interval.

    If actual_boarded is not supplied, boarding is
    estimated from available capacity and queue size.
    """

    def __init__(
        self,
        initial_queue: float = 0
    ):
        if initial_queue < 0:
            raise ValueError(
                "Initial queue cannot be negative."
            )

        self.initial_queue = float(initial_queue)


    # Validate inputs
    @staticmethod
    def _validate_non_negative(
        values,
        name: str
    ):
        values = pd.to_numeric(
            values,
            errors="coerce"
        )

        if values.isna().any():
            raise ValueError(
                f"{name} contains missing or "
                "non-numeric values."
            )

        if (values < 0).any():
            raise ValueError(
                f"{name} cannot contain negative values."
            )

        return values.astype(float)


    # Predict queue
    def predict_queue(
        self,
        estimated_arrivals: pd.DataFrame,
        boarding_capacity: Union[
            float,
            int,
            pd.Series
        ],
        arrivals_col: str = "estimated_arrivals",
        actual_boarded: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate queue size for every interval.

        estimated_arrivals:
            DataFrame with estimated_arrivals column.

        boarding_capacity:
            Scalar capacity or Series containing
            interval-specific boarding capacity.

        actual_boarded:
            Optional actual/predicted boarding counts.

        Returns:
            DataFrame containing:
                estimated_arrivals
                boarding_capacity
                passengers_boarded
                queue_start
                queue_end
                queue_change
                queue_status
        """

        df = estimated_arrivals.copy()


        # Validate arrivals
        if arrivals_col not in df.columns:
            raise ValueError(
                f"Missing arrivals column: {arrivals_col}"
            )

        if df.empty:
            raise ValueError(
                "Arrival forecast is empty."
            )

        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        if df.index.isna().any():
            raise ValueError(
                "Invalid forecast timestamps."
            )

        df = df.sort_index()

        arrivals = self._validate_non_negative(
            df[arrivals_col],
            "Estimated arrivals"
        )


        # Validate boarding capacity
        if np.isscalar(boarding_capacity):

            capacity = pd.Series(
                float(boarding_capacity),
                index=df.index
            )

        else:

            capacity = pd.Series(
                boarding_capacity
            )

            capacity.index = pd.to_datetime(
                capacity.index,
                errors="coerce"
            )

            capacity = capacity.reindex(
                df.index
            )

        capacity = self._validate_non_negative(
            capacity,
            "Boarding capacity"
        )


        # Optional actual boarding
        if actual_boarded is not None:

            boarded_series = pd.Series(
                actual_boarded
            )

            boarded_series.index = pd.to_datetime(
                boarded_series.index,
                errors="coerce"
            )

            boarded_series = (
                boarded_series.reindex(df.index)
            )

            boarded_series = (
                self._validate_non_negative(
                    boarded_series,
                    "Actual boarding"
                )
            )

        else:

            boarded_series = None


        # Queue simulation
        queue_start_values = []
        queue_end_values = []
        boarded_values = []
        queue_changes = []

        current_queue = self.initial_queue

        for timestamp in df.index:

            arrivals_t = float(
                arrivals.loc[timestamp]
            )

            capacity_t = float(
                capacity.loc[timestamp]
            )

            queue_start = current_queue

            # Passengers available for boarding.
            available_passengers = (
                queue_start + arrivals_t
            )

            if boarded_series is None:

                # Estimate actual boarding based
                # on queue and capacity.
                boarded_t = min(
                    available_passengers,
                    capacity_t
                )

            else:

                # Use observed or independently
                # predicted boarding counts.
                boarded_t = float(
                    boarded_series.loc[timestamp]
                )

                # Cannot board more passengers
                # than are available.
                boarded_t = min(
                    boarded_t,
                    available_passengers
                )

                # Cannot exceed capacity.
                boarded_t = min(
                    boarded_t,
                    capacity_t
                )

            # Calculate remaining queue.
            queue_end = max(
                0,
                available_passengers - boarded_t
            )

            # Save interval results.
            queue_start_values.append(
                queue_start
            )

            queue_end_values.append(
                queue_end
            )

            boarded_values.append(
                boarded_t
            )

            queue_changes.append(
                queue_end - queue_start
            )

            # Carry queue into next interval.
            current_queue = queue_end


        # Build result
        result = pd.DataFrame(
            {
                "estimated_arrivals": arrivals,
                "boarding_capacity": capacity,
                "passengers_boarded": boarded_values,
                "queue_start": queue_start_values,
                "queue_end": queue_end_values,
                "queue_change": queue_changes
            },
            index=df.index
        )


        # Queue status
        result["queue_status"] = (
            result["queue_end"]
            .apply(self._queue_status)
        )

        # Summary
        self.summary_ = {
            "initial_queue": self.initial_queue,
            "predicted_peak_queue": float(
                result["queue_end"].max()
            ),
            "peak_time": result[
                "queue_end"
            ].idxmax(),
            "final_queue": float(
                result["queue_end"].iloc[-1]
            ),
            "total_estimated_arrivals": float(
                result[
                    "estimated_arrivals"
                ].sum()
            ),
            "total_predicted_boarded": float(
                result[
                    "passengers_boarded"
                ].sum()
            )
        }

        return result

    
    # Crowd status
    @staticmethod
    def _queue_status(
        queue: float
    ) -> str:

        if queue <= 50:
            return "Low"

        elif queue <= 150:
            return "Moderate"

        elif queue <= 300:
            return "High"

        else:
            return "Critical"

# Summery of the predicted queue results.
    def get_summary(self) -> dict:

        if not hasattr(self, "summary_"):
            raise RuntimeError(
                "Run predict_queue() first."
            )

        return self.summary_
