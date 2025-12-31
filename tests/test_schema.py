import pandas as pd
import pytest

from ml_baseline.schema import InputSchema, validate_and_align


def test_validate_and_align_forbidden_column() -> None:
    schema = InputSchema(
        required_feature_columns=["a"],
        feature_dtypes={"a": "int64"},
        forbidden_columns=["y"],
    )

    df = pd.DataFrame(
        {
            "a": [1, 2],
            "y": [0, 1],  # forbidden (target)
        }
    )

    with pytest.raises(AssertionError, match="Forbidden columns"):
        validate_and_align(df, schema)


def test_validate_and_align_missing_required_column() -> None:
    schema = InputSchema(
        required_feature_columns=["a", "b"],
        feature_dtypes={"a": "int64", "b": "int64"},
    )

    df = pd.DataFrame(
        {
            "a": [1, 2],  # missing "b"
        }
    )

    with pytest.raises(AssertionError, match="Missing required feature columns"):
        validate_and_align(df, schema)
