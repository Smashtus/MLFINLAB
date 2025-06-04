"""
CUSUM Filter for event detection in time series data.

This module provides the CUSUM (Cumulative Sum) filter, a technique used to
detect significant deviations or structural breaks in time series data.
It's often used in finance to sample events for further analysis, such as
labeling for machine learning models.

The filter identifies points where the cumulative sum of differences from a
mean or target value exceeds a predefined threshold.

Usage Example:
--------------
import pandas as pd
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # noqa: E501

# Generate sample price data
np.random.seed(42)
data_points = 200
price_series = pd.Series(
    100 + np.random.randn(data_points).cumsum(),
    index=pd.date_range(start='2023-01-01', periods=data_points, freq='B'),
    name='price'
)

# Define a threshold for the CUSUM filter
# This could be a fixed value or a dynamic series (e.g., based on volatility)  # noqa: E501
# For simplicity, using a fixed threshold here.
fixed_threshold = 2.0

# Apply the CUSUM filter
event_timestamps = cusum_filter(raw_time_series=price_series, threshold=fixed_threshold)

logging.info(f"Original series length: {len(price_series)}")
logging.info(f"Number of events detected by CUSUM filter: {len(event_timestamps)}")  # noqa: E501
logging.info(f"Event timestamps:\n{event_timestamps}")

# Example with a dynamic threshold (e.g., a rolling standard deviation)  # noqa: E501
rolling_std = price_series.pct_change().rolling(window=20).std().bfill() * 5 # Example dynamic threshold  # noqa: E501
dynamic_threshold_series = rolling_std.reindex(price_series.index).fillna(method='bfill').fillna(0.1) # Ensure threshold is positive  # noqa: E501

event_timestamps_dynamic = cusum_filter(raw_time_series=price_series, threshold=dynamic_threshold_series)  # noqa: E501
logging.info(f"Number of events detected with dynamic threshold: {len(event_timestamps_dynamic)}")  # noqa: E501
logging.info(f"Event timestamps (dynamic threshold):\n{event_timestamps_dynamic}")  # noqa: E501
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, List, cast

# Configure basic logging if the module is used standalone
# Note: If imported, the application's logging config would typically take precedence.  # noqa: E501
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # noqa: E501
    )
logger = logging.getLogger(__name__)


def cusum_filter(
    raw_time_series: pd.Series,
    threshold: Union[float, int, pd.Series],
    time_stamps: bool = True,
) -> Union[pd.DatetimeIndex, List[pd.Timestamp]]:
    """
    Applies the Symmetric CUSUM filter to a time series.

    The CUSUM filter is designed to detect shifts in the mean value of a
    time series. It identifies a sequence of upside or downside divergences
    from a reset level (zero). An event is triggered if the cumulative sum
    (S_t) of these divergences reaches a specified `threshold`, at which
    point S_t is reset to 0.

    This implementation can handle both fixed and dynamic (Series-based) thresholds.  # noqa: E501

    Args:
        raw_time_series: Series of data (e.g., close prices, volatility).
                         Must be indexed by timestamps if `time_stamps=True`.
        threshold: When the absolute value of the cumulative sum of changes is  # noqa: E501
                   larger than this threshold, an event is captured.
                   Can be a single float/int (fixed threshold) or a pandas Series  # noqa: E501
                   (dynamic threshold, must share the same index as `raw_time_series`).  # noqa: E501
                   The threshold must be positive.
        time_stamps: If True (default), returns a pd.DatetimeIndex of event times.  # noqa: E501
                     If False, returns a list of pd.Timestamp objects.

    Returns:
        A pd.DatetimeIndex or list of pd.Timestamp objects representing the
        times when events occurred.

    Raises:
        ValueError: If `raw_time_series` is not a pd.Series or is empty.
        ValueError: If `threshold` is non-positive or, if a Series, contains non-positive values  # noqa: E501
                    or does not align with `raw_time_series`.
    """
    if not isinstance(raw_time_series, pd.Series):
        raise ValueError("`raw_time_series` must be a pandas Series.")
    if raw_time_series.empty:
        raise ValueError("`raw_time_series` cannot be empty.")

    # Validate threshold
    if isinstance(threshold, (float, int)):
        if threshold <= 0:
            raise ValueError("Fixed `threshold` must be positive.")
        threshold_series = pd.Series(threshold, index=raw_time_series.index)
    elif isinstance(threshold, pd.Series):
        if not threshold.index.equals(raw_time_series.index):
            logger.warning(
                "Threshold Series index does not match raw_time_series index. Attempting to reindex."  # noqa: E501
            )
            # Attempt to align by reindexing, forward fill then backward fill
            threshold = threshold.reindex(raw_time_series.index, method="ffill")
            threshold = threshold.fillna(
                method="bfill"
            )  # Fill any remaining NaNs at the beginning  # noqa: E501
            if (
                threshold.isnull().any()
            ):  # Check if any NaNs persist after ffill and bfill  # noqa: E501
                raise ValueError(
                    "Reindexed threshold Series still contains NaNs after ffill and bfill. Ensure it covers the time series range."  # noqa: E501
                )
        if (threshold <= 0).any():
            raise ValueError(
                "Dynamic `threshold` Series must contain only positive values."  # noqa: E501
            )
        threshold_series = threshold
    else:
        raise ValueError("`threshold` must be a float, int, or pandas Series.")

    events: List[pd.Timestamp] = []
    s_pos: float = 0.0  # Cumulative sum for positive deviations
    s_neg: float = 0.0  # Cumulative sum for negative deviations

    # Calculate differences once
    diff = raw_time_series.diff().fillna(0)

    for timestamp, price_change, current_threshold in zip(
        diff.index, diff, threshold_series
    ):
        s_pos = max(0, s_pos + price_change)
        s_neg = min(0, s_neg + price_change)

        if s_pos > current_threshold:
            s_pos = 0  # Reset positive sum
            # Ensure timestamp is of correct type if index is not DatetimeIndex
            event_time = cast(pd.Timestamp, timestamp)
            events.append(event_time)

        if s_neg < -current_threshold:  # Negative threshold for negative deviations  # noqa: E501
            s_neg = 0  # Reset negative sum
            event_time = cast(pd.Timestamp, timestamp)
            events.append(event_time)

    if time_stamps:
        if not isinstance(raw_time_series.index, pd.DatetimeIndex):
            logger.warning(
                "raw_time_series index is not a DatetimeIndex. Returning list of indices instead of DatetimeIndex."  # noqa: E501
            )
            return events  # Return as list if index wasn't datetime
        return pd.DatetimeIndex(sorted(list(set(events))))  # Remove duplicates and sort  # noqa: E501
    else:
        return sorted(list(set(events)))


if __name__ == "__main__":
    logger.info("Running cusum.py example...")

    # Generate sample price data
    np.random.seed(42)  # for reproducibility
    data_points = 300
    price_series_main = pd.Series(
        100 + np.random.randn(data_points).cumsum() * 0.5,  # More volatility
        index=pd.date_range(start="2023-01-01", periods=data_points, freq="B"),
        name="price",
    )
    logger.info(f"Sample price series (first 5 points):\n{price_series_main.head()}")  # noqa: E501

    # Example 1: Fixed threshold
    fixed_thresh = 1.5
    logger.info(f"Applying CUSUM filter with fixed threshold: {fixed_thresh}")
    event_ts_fixed = cusum_filter(
        raw_time_series=price_series_main, threshold=fixed_thresh
    )
    logger.info(f"Number of events (fixed threshold): {len(event_ts_fixed)}")
    if len(event_ts_fixed) > 0:
        logger.info(
            f"First 5 event timestamps (fixed threshold):\n{event_ts_fixed[:5]}"
        )

    # Example 2: Dynamic threshold (e.g., based on rolling standard deviation of returns)  # noqa: E501
    logger.info("Applying CUSUM filter with dynamic threshold...")
    # Calculate a dummy dynamic threshold, e.g., 1% of the current price, with a minimum  # noqa: E501
    dynamic_thresh_series = (price_series_main * 0.01).clip(
        lower=0.5
    )  # Threshold is 1% of price, min 0.5  # noqa: E501
    logger.info(
        f"Sample dynamic threshold (first 5 points):\n{dynamic_thresh_series.head()}"  # noqa: E501
    )

    event_ts_dynamic = cusum_filter(
        raw_time_series=price_series_main, threshold=dynamic_thresh_series
    )
    logger.info(f"Number of events (dynamic threshold): {len(event_ts_dynamic)}")  # noqa: E501
    if len(event_ts_dynamic) > 0:
        logger.info(
            f"First 5 event timestamps (dynamic threshold):\n{event_ts_dynamic[:5]}"  # noqa: E501
        )

    # Example 3: Edge case - empty series (should raise ValueError)
    logger.info("Testing with empty series...")
    try:
        cusum_filter(pd.Series([], dtype=float), threshold=1.0)
    except ValueError as e:
        logger.info(f"Caught expected error for empty series: {e}")

    # Example 4: Edge case - non-positive threshold (should raise ValueError)
    logger.info("Testing with non-positive threshold...")
    try:
        cusum_filter(price_series_main, threshold=0)
    except ValueError as e:
        logger.info(f"Caught expected error for non-positive threshold: {e}")
    try:
        cusum_filter(
            price_series_main,
            threshold=pd.Series(
                [-0.5] * len(price_series_main), index=price_series_main.index
            ),
        )
    except ValueError as e:
        logger.info(f"Caught expected error for non-positive dynamic threshold: {e}")  # noqa: E501

    # Example 5: Threshold series with misaligned index (should log warning and attempt reindex)  # noqa: E501
    logger.info("Testing with misaligned dynamic threshold series...")
    misaligned_threshold = pd.Series(
        1.0, index=pd.date_range(start="2022-01-01", periods=data_points, freq="B")
    )  # noqa: E501
    event_ts_misaligned = cusum_filter(
        raw_time_series=price_series_main, threshold=misaligned_threshold
    )  # noqa: E501
    logger.info(
        f"Number of events (misaligned threshold, reindexed): {len(event_ts_misaligned)}"  # noqa: E501
    )

    logger.info("cusum.py example finished.")
