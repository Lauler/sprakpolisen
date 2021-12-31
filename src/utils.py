import pandas as pd
import numpy as np

# Source: https://stackoverflow.com/questions/30045086/pandas-left-join-and-update-existing-column/60572964#60572964
def join_insertion(into_df, from_df, cols, on, by=None, direction=None, mult="error"):
    """
    Suppose A and B are dataframes. A has columns {foo, bar, baz} and B has columns {foo, baz, buz}
    This function allows you to do an operation like:
    "where A and B match via the column foo, insert the values of baz and buz from B into A"
    Note that this'll update A's values for baz and it'll insert buz as a new column.
    This is a lot like DataFrame.update(), but that method annoyingly ignores NaN values in B!

    Optionally, direction can be given as 'backward', 'forward', or nearest to implement a rolling join
    insertion. forward means 'roll into_df values forward to match from_df values', etc. Additionally,
    when doing a rolling join, 'on' should be the roll column and 'by' should be the exact-match columns.
    See pandas.merge_asof() for details.

    Note that 'mult' gets ignored when doing a rolling join. In the case of a rolling join, the first
    appearing record is kept, even if two records match a key from the same distance. Perhaps this
    can be improved...

    :param into_df: dataframe you want to modify
    :param from_df: dataframe with the values you want to insert
    :param cols: list of column names (values to insert)
    :param on: list of column names (values to join on), or a dict of {into:from} column name pairs
    :param by: same format as on; when doing a rolling join insertion, what columns to exact-match on
    :param direction: 'forward', 'backward', or 'nearest'. forward means roll into_df values to match from_df
    :param mult: if a key of into_df matches multiple rows of from_df, how should this be handled?
    an error can be raised, or the first matching value can be inserted, or the last matching value
    can be inserted
    :return: a modified copy of into_df, with updated values using from_df
    """

    # Infer left_on, right_on
    if isinstance(on, dict):
        left_on = list(on.keys())
        right_on = list(on.values())
    elif isinstance(on, list):
        left_on = on
        right_on = on
    elif isinstance(on, str):
        left_on = [on]
        right_on = [on]
    else:
        raise Exception("on should be a list or dictionary")

    # Infer left_by, right_by
    if by is not None:
        if isinstance(by, dict):
            left_by = list(by.keys())
            right_by = list(by.values())
        elif isinstance(by, list):
            left_by = by
            right_by = by
        elif isinstance(by, str):
            left_by = [by]
            right_by = [by]
        else:
            raise Exception("by should be a list or dictionary")
    else:
        left_by = None
        right_by = None

    # Make cols a list if it isn't already
    if isinstance(cols, str):
        cols = [cols]

    # Setup
    A = into_df.copy()
    B = from_df[right_on + cols + ([] if right_by is None else right_by)].copy()

    # Insert row ids
    A["_A_RowId_"] = np.arange(A.shape[0])
    B["_B_RowId_"] = np.arange(B.shape[0])

    # Merge
    if direction is None:
        A = pd.merge(
            left=A,
            right=B,
            how="left",
            left_on=left_on,
            right_on=right_on,
            suffixes=(None, "_y"),
            indicator=True,
        ).sort_values(["_A_RowId_", "_B_RowId_"])

        # Check for rows of A which got duplicated by the merge, and then handle appropriately
        if mult == "error":
            if A.groupby("_A_RowId_").size().max() > 1:
                raise Exception("At least one key of into_df matched multiple rows of from_df.")
        elif mult == "first":
            A = A.groupby("_A_RowId_").first().reset_index()
        elif mult == "last":
            A = A.groupby("_A_RowId_").last().reset_index()

    else:
        A.sort_values(left_on, inplace=True)
        B.sort_values(right_on, inplace=True)
        A = pd.merge_asof(
            left=A,
            right=B,
            direction=direction,
            left_on=left_on,
            right_on=right_on,
            left_by=left_by,
            right_by=right_by,
            suffixes=(None, "_y"),
        ).sort_values(["_A_RowId_", "_B_RowId_"])

    # Insert values from new column(s) into pre-existing column(s)
    mask = A._merge == "both" if direction is None else np.repeat(True, A.shape[0])
    cols_in_both = list(set(into_df.columns.to_list()).intersection(set(cols)))
    for col in cols_in_both:
        A.loc[mask, col] = A.loc[mask, col + "_y"]

    # Drop unwanted columns
    A.drop(
        columns=list(set(A.columns).difference(set(into_df.columns.to_list() + cols))),
        inplace=True,
    )

    return A
