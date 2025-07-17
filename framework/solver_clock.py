import time
import logging
from typing import Optional

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


class SolverClock:
    """
    This class keeps track of the time across a solver execution
    """

    def __init__(
        self,
        run_time_limit: float,
        start_time: Optional[float] = None,
    ):
        self.run_time_limit = run_time_limit
        self.start_time = start_time
        self.paused_intervals = []
        self.current_pause_start = None

        if start_time is not None:
            logging.info(f"Solver clock has been started")

    def start(self):
        """Start the timer."""
        if self.start_time is None:
            logging.info(f"Start Solver Clock")
            self.start_time = time.time()
            self.last_log = self.start_time
        else:
            ValueError(f"Timer has already been started")

    def pause(self):
        """Pause the timer."""
        if self.current_pause_start is None:
            logging.info(f"Pause Solver Clock")
            self.current_pause_start = time.time()
        else:
            raise ValueError("Timer is already paused")

    def resume(self):
        """Resume the timer."""
        if self.current_pause_start is not None:
            logging.info(f"Resume Solver Clock")
            pause_end = time.time()
            self.paused_intervals.append((self.current_pause_start, pause_end))
            self.current_pause_start = None
        else:
            ValueError(
                "Timer is still running - resuming does not make sense here"
            )

    def get_total_paused_time(self) -> float:
        """Calculate the total time the timer was paused."""
        return sum(end - start for start, end in self.paused_intervals)

    def get_elapsed_time(self) -> float:
        """Get the elapsed time since the timer was started, excluding paused intervals."""
        return time.time() - self.start_time - self.get_total_paused_time()

    def get_remaining_time(self) -> float:
        """Get the remaining execution time."""
        return max(0, self.run_time_limit - self.get_elapsed_time())

