"""
Triple Barrier Method for labeling financial data.

This module provides functions for implementing the triple-barrier method,
a technique used in financial machine learning to label events based on price
movements.

Usage Example:
--------------
# (Provide a more concrete example later)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample data
close_prices = pd.Series(np.random.rand(100) * 100,
                         index=pd.date_range(start='2023-01-01', periods=100, freq='B'))
events_idx = close_prices.iloc[[10, 30, 50, 70, 90]].index
daily_vol = pd.Series(np.random.rand(100) * 0.02 + 0.01, index=close_prices.index) # Target volatility  # noqa: E501
pt_sl_levels = [1, 1] # Profit taking and stop loss multipliers for target volatility  # noqa: E501
min_return = 0.005 # Minimum return for triple barrier
num_threads = 1 # Number of threads for processing

# 1. Add vertical barrier
vertical_barriers = add_vertical_barrier(t_events=events_idx, close=close_prices, num_days=5)

# 2. Apply profit taking and stop loss on t1 (first touch time)
# This function is usually called internally by get_events,
# but can be used standalone if needed.
# For standalone usage, molecule would be a subset of events_idx.
# first_touch_times_df = apply_pt_sl_on_t1(close=close_prices,  # noqa: E501
#                                       events=events_idx,  # noqa: E501
#                                       pt_sl=pt_sl_levels,  # noqa: E501
#                                       molecule=events_idx[:2]) # Example with first two events

# 3. Get events (first touch times and side if applicable)
triple_barrier_events = get_events(close=close_prices,
                                   t_events=events_idx,
                                   pt_sl=pt_sl_levels,
                                   target=daily_vol,
                                   min_ret=min_return,
                                   num_threads=num_threads,
                                   vertical_barrier_times=vertical_barriers)

# 4. Get bins (labels)
# labels = get_bins(triple_barrier_events=triple_barrier_events, close=close_prices)

# 5. Drop rare labels (if necessary)
# filtered_labels = drop_labels(events=labels, min_pct=0.05)

# print("Triple Barrier Events:\n", triple_barrier_events)
# print("\nLabels:\n", labels)
# print("\nFiltered Labels:\n", filtered_labels)
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Union, Optional

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def add_vertical_barrier(
    t_events: pd.Series,
    close: pd.Series,
    num_days: int = 0,
    num_hours: int = 0,
    num_minutes: int = 0,
    num_seconds: int = 0,
) -> pd.Series:
    """
    Adds a vertical barrier for each event.

    For each index in t_events, finds timestamp of next price bar at or after
    a specified timedelta. Used as max holding period.

    Args:
        t_events: Series of timestamps for event starts (e.g., CUSUM trigger).  # noqa: E501
        close: Series of close prices, indexed by timestamps.  # noqa: E501
        num_days: Number of days for vertical barrier.  # noqa: E501
        num_hours: Number of hours for vertical barrier.  # noqa: E501
        num_minutes: Number of minutes for vertical barrier.  # noqa: E501
        num_seconds: Number of seconds for vertical barrier.  # noqa: E501

    Returns:
        Series of timestamps when each vertical barrier is reached.  # noqa: E501
        Index matches `t_events`. pd.NaT if barrier cannot be formed.  # noqa: E501
    """
    timedelta_val = pd.Timedelta(
        days=num_days, hours=num_hours, minutes=num_minutes, seconds=num_seconds
    )  # noqa: E501
    if timedelta_val <= pd.Timedelta(0):
        logging.warning(
            "Vertical barrier timedelta must be positive. Returning NaT for all events."  # noqa: E501
        )
        return pd.Series(pd.NaT, index=t_events.index)

    # Find the union of all timestamps that could be potential barrier times  # noqa: E501
    relevant_indices = close.index.searchsorted(t_events + timedelta_val, side="left")  # noqa: E501

    # Ensure indices are within bounds
    relevant_indices = np.clip(relevant_indices, 0, len(close.index) - 1)  # noqa: E501

    # Removed unused 'barrier_times' variable here

    # For events where calculated barrier time is before/at event time itself  # noqa: E501
    # (e.g. insufficient future data or timedelta too small vs data frequency),  # noqa: E501
    # or if timedelta pushes it beyond available data, set to NaT.  # noqa: E501
    # Also handles t_events + timedelta_val > last close.index.

    # Iterate and find the correct barrier time for each event
    final_barrier_times = []
    for event_idx in t_events.index:
        event_time = t_events.loc[event_idx]
        target_time = event_time + timedelta_val

        # Find the first index in close.index that is >= target_time
        future_prices_idx = close.index.searchsorted(target_time, side="left")

        if future_prices_idx < len(close.index):
            final_barrier_times.append(close.index[future_prices_idx])
        else:
            # If target_time is beyond the last timestamp in close.index
            final_barrier_times.append(pd.NaT)

    return pd.Series(final_barrier_times, index=t_events.index)


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: List[Union[float, int]],
    molecule: pd.Index,
) -> pd.DataFrame:
    """
    Applies profit-taking (PT) and stop-loss (SL) limits to determine the
    first barrier touch time.

    This function processes a subset of events (defined by `molecule`) and
    identifies whether the price path first hits PT, SL, or vertical
    barrier (t1).

    Args:
        close: Series of close prices, indexed by timestamps.
        events: DataFrame with event information:
            - index: Event start times.
            - 't1': Timestamps of vertical barriers.
            - 'trgt': Target return (e.g., volatility) for setting PT/SL levels.  # noqa: E501
            - 'side' (optional): Position side (1 for long, -1 for short).  # noqa: E501
                                If not provided, barriers are set symmetrically.
        pt_sl: List/array of two non-negative floats:  # noqa: E501
            - pt_sl[0]: Profit-taking multiplier.
            - pt_sl[1]: Stop-loss multiplier.
            Value of 0 means barrier is disabled.
        molecule: A pd.Index subset of `events.index` to be processed.

    Returns:
        DataFrame with ['t1', 'pt', 'sl'] cols, index matching `molecule`.  # noqa: E501
        - 't1': Timestamp of first barrier touch.
        - 'pt': Profit-taking level if touched, else NaN.
        - 'sl': Stop-loss level if touched, else NaN.
        't1' in output is time of *first* barrier touch (PT, SL, or vertical).  # noqa: E501
    """
    # Subset events to the specific molecule
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)  # t1 is the vertical barrier time

    profit_take_mult = float(pt_sl[0])
    stop_loss_mult = float(pt_sl[1])

    if "side" in events.columns:
        long_barriers = profit_take_mult > 0
        short_barriers = (
            profit_take_mult > 0
        )  # Assuming symmetric for now if side is present  # noqa: E501
    else:  # Symmetric barriers if no side is specified
        long_barriers = profit_take_mult > 0
        short_barriers = (
            stop_loss_mult > 0
        )  # Corrected: use stop_loss_mult for short side target  # noqa: E501

    for loc, vertical_barrier_time in events_["t1"].fillna(close.index[-1]).items():  # noqa: E501
        path_prices = close[
            loc:vertical_barrier_time
        ]  # Prices from event start to vertical barrier  # noqa: E501
        if (
            path_prices.empty
        ):  # Handle event start beyond close data or too soon VB  # noqa: E501
            out.loc[loc, "t1"] = vertical_barrier_time  # Or pd.NaT
            out.loc[loc, "pt_level"] = np.nan
            out.loc[loc, "sl_level"] = np.nan
            logging.debug(
                f"No price path for event {loc}, vb_time: {vertical_barrier_time}. Skip."  # noqa: E501
            )
            continue

        entry_price = path_prices.iloc[0]
        target_ret = events_.at[loc, "trgt"]

        if long_barriers:
            pt_level = entry_price * (1 + target_ret * profit_take_mult)  # noqa: E501
        else:
            pt_level = np.nan  # No profit taking if multiplier is 0 or side not allow

        if (
            short_barriers
        ):  # Use stop_loss_mult for the lower barrier's width  # noqa: E501
            sl_level = entry_price * (1 - target_ret * stop_loss_mult)  # noqa: E501
        else:
            sl_level = np.nan  # No stop loss if multiplier is 0 or side not allow

        # Handle cases where side is present
        if "side" in events.columns:
            side = events_.at[loc, "side"]
            if side == 1:  # Long position
                if profit_take_mult > 0:
                    pt_level = entry_price * (
                        1 + target_ret * profit_take_mult
                    )  # noqa: E501
                else:
                    pt_level = np.nan
                if stop_loss_mult > 0:
                    sl_level = entry_price * (
                        1 - target_ret * stop_loss_mult
                    )  # noqa: E501
                else:
                    sl_level = np.nan
            elif side == -1:  # Short position
                if profit_take_mult > 0:  # For short, PT is downwards
                    pt_level = entry_price * (
                        1 - target_ret * profit_take_mult
                    )  # noqa: E501
                else:
                    pt_level = np.nan
                if stop_loss_mult > 0:  # For short, SL is upwards
                    sl_level = entry_price * (
                        1 + target_ret * stop_loss_mult
                    )  # noqa: E501
                else:
                    sl_level = np.nan
            else:  # Neutral side, no barriers
                pt_level = np.nan
                sl_level = np.nan

        first_touch_time = vertical_barrier_time

        for t, price in path_prices.iloc[
            1:
        ].items():  # Skip the entry price itself  # noqa: E501
            touched_pt = False
            touched_sl = False

            # Check profit take
            if pd.notna(pt_level):
                if (
                    "side" in events.columns and events_.at[loc, "side"] == -1
                ):  # Short position
                    if price <= pt_level:
                        touched_pt = True
                elif price >= pt_level:  # Long or symmetric
                    touched_pt = True

            # Check stop loss
            if pd.notna(sl_level):
                if (
                    "side" in events.columns and events_.at[loc, "side"] == -1
                ):  # Short position
                    if price >= sl_level:
                        touched_sl = True
                elif price <= sl_level:  # Long or symmetric
                    touched_sl = True

            if touched_pt or touched_sl:
                first_touch_time = t
                break  # Exit loop once a barrier is touched

        out.loc[loc, "t1"] = first_touch_time
        # Store the actual levels for clarity, though not strictly in original snippet
        out.loc[loc, "pt_level"] = pt_level
        out.loc[loc, "sl_level"] = sl_level

    return out


def get_events(
    close: pd.Series,
    t_events: pd.Index,
    pt_sl: List[Union[float, int]],
    target: pd.Series,
    min_ret: float,
    num_threads: int,  # Kept for signature compatibility, not used without multi-processing  # noqa: E501
    vertical_barrier_times: Optional[pd.Series] = None,
    side_prediction: Optional[pd.Series] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Orchestrates the triple-barrier method to determine event outcomes.

    Identifies time of first barrier touch (PT, SL, or vertical)
    for each event initiated at `t_events`.

    Args:
        close: Series of close prices, indexed by timestamps.
        t_events: Index of timestamps that seed each triple barrier (e.g., CUSUM).  # noqa: E501
        pt_sl: List/array of two non-negative floats:  # noqa: E501
            - pt_sl[0]: Profit-taking multiplier for `target`.
            - pt_sl[1]: Stop-loss multiplier for `target`.
            0 means respective horizontal barrier disabled.
        target: Series (e.g., daily volatility) used with `pt_sl` to set  # noqa: E501
                width of horizontal barriers. Must be positive.
        min_ret: Minimum target return for running a triple-barrier search.  # noqa: E501
                 Events with `target` < `min_ret` are filtered out.
        num_threads: Number of threads (currently not implemented, single-threaded). # noqa: E501
        vertical_barrier_times: Series of vertical barrier timestamps. If None/False, # noqa: E501
                                VBs disabled (events last indefinitely or until  # noqa: E501
                                H-barrier hit; usually combined with  # noqa: E501
                                `add_vertical_barrier`).
        side_prediction: Optional Series for side of bet (1 long, -1 short)  # noqa: E501
                         from primary model (for meta-labeling).
        verbose: If True, logs information about the process.

    Returns:
        DataFrame where:
        - Index: Event start times.
        - 't1': Timestamp of the first barrier touch (PT, SL, or vertical).  # noqa: E501
        - 'trgt': The target return used for the event.
        - 'side' (optional): The predicted side of the position.
        - 'pt': Profit-taking multiplier used.
        - 'sl': Stop-loss multiplier used.
    """
    if verbose:
        logging.info("Starting get_events process.")

    # 1. Filter events by min_ret and target positivity
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]
    if target.empty:
        if verbose:
            logging.warning(
                "No events meet min_ret or target is empty/all non-positive."  # noqa: E501
            )
        return pd.DataFrame()

    t_events = target.index

    # 2. Create events DataFrame
    events = pd.DataFrame(index=t_events)
    events["trgt"] = target

    # 3. Add vertical barrier times if provided
    if vertical_barrier_times is not False and vertical_barrier_times is not None:
        events["t1"] = vertical_barrier_times.reindex(events.index).fillna(
            close.index[-1]
        )  # noqa: E501
    else:  # No vertical barrier, events could run indefinitely (or until close.index[-1])  # noqa: E501
        events["t1"] = pd.Series(close.index[-1], index=t_events)

    # 4. Add side prediction if provided
    if side_prediction is not None:
        events["side"] = side_prediction.reindex(events.index).fillna(
            0
        )  # Default to neutral if no side

    # Store pt/sl multipliers
    events["pt"] = float(pt_sl[0])
    events["sl"] = float(pt_sl[1])

    # Filter out events where vertical barrier is before or at event start time
    # Also filter out events where the vertical barrier itself is NaT
    events = events[events["t1"].notna() & (events["t1"] > events.index)]
    if events.empty:
        if verbose:
            logging.warning("No valid events after VB check (all VBs too soon or NaT).")  # noqa: E501
        return pd.DataFrame()

    # 5. Apply profit-take and stop-loss (single-threaded for now)
    # Original used multiprocessing (mp_pandas_obj).
    # Simplified here to single-threaded.
    # Performance-critical apps might need careful multiprocessing re-add
    # without mlfinlab's specific utilities.

    # 'molecule' in original code was for parallel processing.
    # Here, we process all valid events.
    df_first_touch = apply_pt_sl_on_t1(
        close=close, events=events, pt_sl=pt_sl, molecule=events.index
    )  # noqa: E501

    # Update events['t1'] with the actual first touch times and pt/sl levels
    events["t1"] = df_first_touch["t1"]
    if "pt_level" in df_first_touch.columns:
        events["pt_level"] = df_first_touch["pt_level"]
    if "sl_level" in df_first_touch.columns:
        events["sl_level"] = df_first_touch["sl_level"]

    # Remove events that did not touch any barrier (should not happen if t1 is always set)  # noqa: E501
    # and events where target was NaN (already filtered by target > min_ret)
    # Also remove events where t1 became NaT after apply_pt_sl_on_t1
    events = events.dropna(subset=["t1"])

    if verbose:
        logging.info(f"Finished get_events. Found {len(events)} events.")
    return events


def get_bins(triple_barrier_events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Computes labels (bins) for triple-barrier events.

    This function determines the outcome of each event:
    - If 'side' is NOT in `triple_barrier_events`:
        - Label 1 if PT hit first & price increased.
        - Label -1 if SL hit first & price decreased.
        - Label by price movement vs entry if VB hit (1 if price > entry, etc.).
    - If 'side' IS in `triple_barrier_events` (meta-labeling):
        - Label 1 if side bet profitable (e.g., long and price up).
        - Label 0 if side bet not profitable.

    Args:
        triple_barrier_events: DataFrame from `get_events`, containing:
            - index: Event start times.
            - 't1': Timestamp of first barrier touch.
            - 'trgt': Target return.
            - 'side' (optional): Predicted side of position.
            - 'pt_level' (optional, from modified apply_pt_sl_on_t1): PT price.
            - 'sl_level' (optional, from modified apply_pt_sl_on_t1): SL price.
            - '_original_vbt' (optional): Original VB time before PT/SL adjust.
        close: Series of close prices, indexed by timestamps.

    Returns:
        DataFrame with event outcomes, including:
        - 'ret': Return of the event.
        - 'bin': The label ({1, -1, 0} for price action, or {1, 0} for meta-labeling).  # noqa: E501
    """
    if triple_barrier_events.empty:
        logging.warning("get_bins received an empty triple_barrier_events DataFrame.")  # noqa: E501
        return pd.DataFrame(columns=["ret", "bin"])

    # 1. Calculate returns for each event
    entry_prices = close.reindex(triple_barrier_events.index)

    # Ensure exit times valid & within close.index bounds.
    # Handle NaT in t1: map to point where close price can be fetched
    # (e.g. last known) or drop/ignore events earlier.
    # Reindex produces NaN for NaT t1, handled below.
    exit_prices_values = close.reindex(triple_barrier_events["t1"]).values  # noqa: E501
    exit_prices_s = pd.Series(exit_prices_values, index=triple_barrier_events.index)  # noqa: E501

    # Prevent div by zero or NaN issues if entry_price missing for an event
    valid_indices = entry_prices.notna() & exit_prices_s.notna()  # noqa: E501
    if not valid_indices.all():
        logging.warning(
            f"{(~valid_indices).sum()} events miss entry/exit prices (due to NaT t1?). Returns NaN."  # noqa: E501
        )

    out = pd.DataFrame(index=triple_barrier_events.index)
    out["ret"] = np.nan  # Initialize with NaN
    out.loc[valid_indices, "ret"] = (
        exit_prices_s[valid_indices] / entry_prices[valid_indices]
    ) - 1.0  # noqa: E501

    if "pt_level" in triple_barrier_events.columns:
        out["pt_level"] = triple_barrier_events["pt_level"]
    if "sl_level" in triple_barrier_events.columns:
        out["sl_level"] = triple_barrier_events["sl_level"]
    out["t1"] = triple_barrier_events["t1"]

    # Initialize bins
    out["bin"] = 0

    for idx, row in triple_barrier_events.iterrows():
        if not valid_indices.get(
            idx, False
        ):  # Skip if no valid return (e.g. t1 was NaT)
            out.loc[idx, "bin"] = 0
            logging.debug(
                f"Skipping bin assignment for event {idx} due to invalid return."
            )
            continue

        entry_price = entry_prices[idx]
        exit_price = exit_prices_s[idx]

        # Case 1: Meta-labeling (side is present and not 0)
        if "side" in row and row["side"] != 0:
            side = row["side"]
            # For meta-labeling, profit is simply if the return was in the direction of the bet  # noqa: E501
            if (side == 1 and out.loc[idx, "ret"] > 0) or (
                side == -1 and out.loc[idx, "ret"] < 0
            ):
                out.loc[idx, "bin"] = 1
            else:
                out.loc[idx, "bin"] = 0  # Bet was not profitable or return was zero

        # Case 2: Labeling by price action (no side or side is 0)
        else:
            pt_level = row.get("pt_level", np.nan)
            sl_level = row.get("sl_level", np.nan)

            # Determine if PT or SL was hit based on exit price and the levels
            # This assumes that if a horizontal barrier was hit, 't1' is that touch time,
            # and 'exit_price' is the price at that 't1'.

            pt_hit = False
            if pd.notna(pt_level):
                # For long/symmetric, PT is hit if price goes up to pt_level
                # For short, PT is hit if price goes down to pt_level
                is_short_side = "side" in row and row["side"] == -1
                if is_short_side:
                    if exit_price <= pt_level:
                        pt_hit = True
                else:  # Long or symmetric
                    if exit_price >= pt_level:
                        pt_hit = True

            sl_hit = False
            if pd.notna(sl_level):
                is_short_side = "side" in row and row["side"] == -1
                if is_short_side:
                    if exit_price >= sl_level:
                        sl_hit = True
                else:  # Long or symmetric
                    if exit_price <= sl_level:
                        sl_hit = True

            # Check if the touch time 't1' is earlier than the original vertical barrier time.  # noqa: E501
            # Strong indicator H-barrier hit if t1 < original_vb_time.
            # Requires '_original_vbt' in triple_barrier_events.
            original_vb_time = row.get("_original_vbt", pd.NaT)
            horizontal_hit_implied = (
                pd.notna(row["t1"])
                and pd.notna(original_vb_time)
                and row["t1"] < original_vb_time
            )

            # Disambiguation: exit_price exactly a barrier level = hit.
            # Exit_price beyond a barrier = hit.
            # t1 < original_vb_time implies H-barrier hit.

            # Priority:
            # 1. PT hit, SL not: bin = 1
            # 2. SL hit, PT not: bin = -1
            # 3. Both hit (e.g. gap through, levels very close):
            #    - ret > 0: bin = 1; ret < 0: bin = -1; else: bin = 0
            # 4. Neither hit (vertical barrier): Based on return direction.

            # More precise for non-meta:
            # Sign of return determines label if H-barrier hit.  # noqa: E501
            # PT hit: ret > 0 (long/symm) or ret < 0 (short).  # noqa: E501
            # SL hit: ret < 0 (long/symm) or ret > 0 (short).  # noqa: E501

            if horizontal_hit_implied:
                if pt_hit and not sl_hit:  # PT hit
                    out.loc[idx, "bin"] = (
                        1 if not ("side" in row and row["side"] == -1) else -1
                    )  # if short, PT makes ret < 0  # noqa: E501
                elif sl_hit and not pt_hit:  # SL hit
                    out.loc[idx, "bin"] = (
                        -1 if not ("side" in row and row["side"] == -1) else 1
                    )  # if short, SL makes ret > 0  # noqa: E501
                elif pt_hit and sl_hit:  # Both hit (ambiguous)
                    # Default to sign of return. Needs careful thought.  # noqa: E501
                    # Non-meta: usually closer one or rule. Price gaps complicate.  # noqa: E501
                    # Assume: if ret > 0 -> 1, if ret < 0 -> -1  # noqa: E501
                    if out.loc[idx, "ret"] > 0:
                        out.loc[idx, "bin"] = 1
                    elif out.loc[idx, "ret"] < 0:
                        out.loc[idx, "bin"] = -1
                    else:
                        out.loc[idx, "bin"] = 0
                else:  # H-implied, but levels don't match exit (gaps/discrete data)  # noqa: E501
                    # Fallback to sign of return if H-hit strongly implied by time.  # noqa: E501
                    if out.loc[idx, "ret"] > 0:
                        out.loc[idx, "bin"] = 1
                    elif out.loc[idx, "ret"] < 0:
                        out.loc[idx, "bin"] = -1
                    else:
                        out.loc[idx, "bin"] = 0
            else:  # Vertical barrier hit
                if out.loc[idx, "ret"] > 0:
                    out.loc[idx, "bin"] = 1
                elif out.loc[idx, "ret"] < 0:
                    out.loc[idx, "bin"] = -1
                else:
                    out.loc[idx, "bin"] = 0

    return out


def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """
    Recursively drops rare labels from the events DataFrame.

    This function inspects the 'bin' column (labels) and removes events
    belonging to classes that represent less than `min_pct` of the total events.
    The process is repeated until all remaining classes meet the `min_pct` threshold.

    Args:
        events: DataFrame, typically the output of `get_bins`, containing a 'bin' column.
        min_pct: Minimum percentage for a label class to be retained.
                 Value should be between 0 and 1.

    Returns:
        DataFrame with rare labels removed. If all labels are dropped,
        an empty DataFrame is returned.
    """
    if "bin" not in events.columns:
        logging.error("Column 'bin' not found in events DataFrame. Cannot drop labels.")  # noqa: E501
        return events
    if events.empty:
        logging.info("Input events DataFrame is empty. Returning as is.")
        return events

    current_events = events.copy()  # Work on a copy

    while True:
        if (
            current_events.empty
        ):  # Stop if all events were dropped in previous iterations
            logging.warning("All labels dropped due to min_pct threshold.")
            break

        df_counts = current_events["bin"].value_counts(normalize=True)

        if df_counts.empty:  # No labels left to count
            break

        min_freq = df_counts.min()

        if min_freq >= min_pct:  # Stop if smallest class is frequent enough
            break

        rarest_label = df_counts.idxmin()  # Label of the rarest class
        current_events = current_events[current_events["bin"] != rarest_label]
        logging.debug(
            f"Dropped label {rarest_label} (freq {min_freq:.4f} < {min_pct}). Left: {len(current_events)}"  # noqa: E501
        )

    if (
        current_events.empty and not events.empty
    ):  # All rows dropped from originally non-empty df
        return pd.DataFrame(
            columns=events.columns
        )  # Return empty df with original columns

    return current_events


if __name__ == "__main__":
    # Setup basic logging for the __main__ example
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Running triple_barrier.py example...")

    # Generate sample data
    np.random.seed(42)  # for reproducibility
    num_points = 200
    close_prices = pd.Series(
        100 + np.random.randn(num_points).cumsum(),
        index=pd.date_range(start="2023-01-01", periods=num_points, freq="B"),
        name="close",
    )

    # Simulate some event triggers (e.g., from a CUSUM filter)
    event_indices = [10, 30, 55, 80, 100, 120, 150, 180]
    # Ensure event indices are valid for the close_prices Series
    event_indices = [i for i in event_indices if i < len(close_prices)]
    t_events = close_prices.index[event_indices]

    # Calculate daily volatility as target for barrier width
    daily_vol = (
        close_prices.pct_change().rolling(window=20).std().fillna(0.01)
    )  # Daily vol
    # daily_vol = daily_vol * np.sqrt(252)  # Annualized - decide target daily/annualized  # noqa: E501
    daily_vol = (
        daily_vol.replace(0, 0.01).shift(1).fillna(0.01)
    )  # Shift to avoid lookahead, ensure positive  # noqa: E501

    # Parameters for triple barrier
    pt_sl_multipliers = [1, 1]  # PT x1 target_vol, SL x1 target_vol  # noqa: E501
    min_event_return = 0.0001  # Minimum target_vol for an event  # noqa: E501
    holding_days = 7  # Max holding period for an event

    logging.info(f"Number of close prices: {len(close_prices)}")  # noqa: E501
    logging.info(f"Number of events: {len(t_events)}")
    logging.info(f"Close prices head:\n{close_prices.head()}")
    logging.info(
        f"Event timestamps (t_events) head:\n{t_events[:5] if len(t_events) > 0 else 'No events'}"  # noqa: E501
    )
    logging.info(f"Daily volatility (target) head:\n{daily_vol.head()}")  # noqa: E501

    if len(t_events) == 0:
        logging.error("No event timestamps generated. Exiting example.")
    else:
        # 1. Add vertical barrier
        logging.info(f"Adding vertical barrier with num_days={holding_days}...")  # noqa: E501
        # add_vertical_barrier expects Series for t_events (values=timestamps, index=event id)  # noqa: E501
        # If t_events is Index of timestamps, convert appropriately
        t_events_series = pd.Series(t_events, index=t_events)
        vertical_barriers = add_vertical_barrier(
            t_events=t_events_series, close=close_prices, num_days=holding_days
        )  # noqa: E501
        logging.info(f"Vertical barriers head:\n{vertical_barriers.head()}")  # noqa: E501

        # Store original vertical barrier times for get_bins logic
        _original_vbt = vertical_barriers.copy()

        # 2. Get events (first touch times, side, etc.)
        logging.info("Calculating triple barrier events...")
        triple_barrier_events_df = get_events(
            close=close_prices,
            t_events=t_events,  # get_events expects an Index for t_events
            pt_sl=pt_sl_multipliers,
            target=daily_vol,  # target should be aligned with t_events
            min_ret=min_event_return,
            num_threads=1,
            vertical_barrier_times=vertical_barriers,
            verbose=True,
        )  # Pass verbose

        if triple_barrier_events_df.empty:
            logging.error(
                "No triple barrier events generated by get_events. Exiting."
            )  # noqa: E501
        else:
            logging.info(
                f"Generated triple barrier events:\n{triple_barrier_events_df.head()}"  # noqa: E501
            )

            # Add original vertical barrier times to use in get_bins
            triple_barrier_events_df["_original_vbt"] = _original_vbt.reindex(
                triple_barrier_events_df.index
            )  # noqa: E501

            # 3. Get bins (labels for the events)
            logging.info("Calculating bins (labels)...")
            # get_bins uses 'pt_level' and 'sl_level' from events_df if available. # noqa: E501
            # get_events was modified to include these.

            labels_df = get_bins(
                triple_barrier_events=triple_barrier_events_df, close=close_prices
            )  # noqa: E501
            logging.info(f"Generated labels (head):\n{labels_df.head()}")  # noqa: E501
            if not labels_df.empty:
                logging.info(
                    f"Label distribution before filtering:\n{labels_df['bin'].value_counts(normalize=True)}"  # noqa: E501
                )

            # 4. Drop rare labels (optional)
            min_occurrence_pct = 0.1
            logging.info(
                f"Dropping labels with occurrence < {min_occurrence_pct*100}%..."
            )  # noqa: E501
            filtered_labels_df = drop_labels(
                events=labels_df.copy(), min_pct=min_occurrence_pct
            )  # noqa: E501

            if not filtered_labels_df.empty:
                logging.info(
                    f"Filtered labels (head):\n{filtered_labels_df.head()}"
                )  # noqa: E501
                logging.info(
                    f"Label distribution after filtering:\n{filtered_labels_df['bin'].value_counts(normalize=True)}"  # noqa: E501
                )
            else:
                logging.info(
                    "No labels left after filtering or initial labels_df was empty."
                )  # noqa: E501

            logging.info("Triple_barrier.py example finished.")
