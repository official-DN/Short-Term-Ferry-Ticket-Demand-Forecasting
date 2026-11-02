from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class WaitingTimeConfig:
    """
    Configuration for passenger waiting time estimation.

    boarding_rate:
        Effective passengers boarded per minute.

    service_mode:
        "continuous" for continuous boarding service.
        "scheduled" for boarding at ferry departure times.

    ferry_capacity:
        Maximum passengers boarded by one ferry.

    ferry_interval_minutes:
        Time between ferry departures, in minutes.
    """

    boarding_rate: float = 10.0
    service_mode: str = "continuous"

    ferry_capacity: int = 100
    ferry_interval_minutes: int = 30


class WaitingTimeEstimator:

    def __init__(
        self,
        config: Optional[WaitingTimeConfig] = None
    ):
        self.config = config or WaitingTimeConfig()

        if self.config.boarding_rate <= 0:
            raise ValueError(
                "Boarding rate must be greater than zero."
            )

        if self.config.service_mode not in [
            "continuous",
            "scheduled"
        ]:
            raise ValueError(
                "service_mode must be 'continuous' or 'scheduled'."
            )

        if self.config.ferry_capacity <= 0:
            raise ValueError(
                "Ferry capacity must be greater than zero."
            )

        if self.config.ferry_interval_minutes <= 0:
            raise ValueError(
                "Ferry interval must be greater than zero."
            )

    def estimate_continuous_wait(self, queue_size):
        """
        Estimate waiting time using a continuous boarding rate.

        Formula:
            Waiting time = Queue size / Boarding rate
        """

        if queue_size < 0:
            raise ValueError(
                "Queue size cannot be negative."
            )

        wait_minutes = (
            queue_size / self.config.boarding_rate
        )

        return wait_minutes

    def estimate_scheduled_wait(
        self,
        queue_size,
        minutes_until_next_ferry: float = 0
    ):
        """
        Estimate waiting time using scheduled ferry departures.

        Assumes:
        - Passengers board in batches.
        - Each ferry boards up to ferry_capacity passengers.
        - Ferries depart at regular intervals.
        """

        if queue_size < 0:
            raise ValueError(
                "Queue size cannot be negative."
            )

        if minutes_until_next_ferry < 0:
            raise ValueError(
                "Minutes until next ferry cannot be negative."
            )

        if queue_size == 0:
            return 0.0

        capacity = self.config.ferry_capacity
        interval = self.config.ferry_interval_minutes

        # Number of ferry departures needed to serve the queue
        ferries_needed = (
            queue_size + capacity - 1
        ) // capacity

        # Time to the final ferry departure
        wait_minutes = (
            minutes_until_next_ferry
            + (ferries_needed - 1) * interval
        )

        return float(wait_minutes)

    def estimate_waiting_time(
        self,
        queue_size,
        minutes_until_next_ferry: float = 0
    ):
        """
        Estimate waiting time based on configured service mode.
        """

        if self.config.service_mode == "continuous":

            return self.estimate_continuous_wait(
                queue_size
            )

        return self.estimate_scheduled_wait(
            queue_size,
            minutes_until_next_ferry
        )

    def analyze(
        self,
        crowd_data: pd.DataFrame,
        queue_col: str = "queue_end",
        minutes_until_next_ferry_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Estimate waiting time for each predicted queue interval.

        Parameters
        ----------
        crowd_data:
            DataFrame containing predicted queue information.

        queue_col:
            Column containing estimated queue size.

        minutes_until_next_ferry_col:
            Optional column containing minutes until the next
            ferry departure.

        Returns
        -------
        DataFrame:
            Original data plus estimated waiting time.
        """

        if queue_col not in crowd_data.columns:
            raise ValueError(
                f"Missing required queue column: {queue_col}"
            )

        result = crowd_data.copy()

        waiting_times = []

        for _, row in result.iterrows():

            queue_size = float(row[queue_col])

            if (
                self.config.service_mode == "scheduled"
                and minutes_until_next_ferry_col is not None
            ):

                minutes_until_next_ferry = float(
                    row[minutes_until_next_ferry_col]
                )

            else:
                minutes_until_next_ferry = 0

            wait = self.estimate_waiting_time(
                queue_size,
                minutes_until_next_ferry
            )

            waiting_times.append(wait)

        result["estimated_wait_minutes"] = waiting_times

        # Convert waiting time into readable format
        result["estimated_wait_readable"] = (
            result["estimated_wait_minutes"].apply(
                self.format_waiting_time
            )
        )

        return result

    @staticmethod
    def format_waiting_time(minutes):
        """
        Convert minutes into readable hours and minutes.
        """

        minutes = max(0, int(round(minutes)))

        hours = minutes // 60
        remaining_minutes = minutes % 60

        if hours > 0:
            return (
                f"{hours} hour(s) "
                f"{remaining_minutes} minute(s)"
            )

        return f"{remaining_minutes} minute(s)"
