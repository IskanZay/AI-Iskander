from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


def test_compute_quality_flags_constant_columns():
    """Тестирует новую эвристику has_constant_columns."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "constant_col": [10, 10, 10, 10],  
        "normal_col": ["A", "B", "C", "D"],
        "numeric_col": [1.1, 2.2, 3.3, 4.4]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_constant_columns"] is True
    
    assert "constant_col" in flags["constant_columns"]
    
    assert "id" not in flags["constant_columns"]
    assert "normal_col" not in flags["constant_columns"]
    assert "numeric_col" not in flags["constant_columns"]


def test_compute_quality_flags_high_cardinality():
    """Тестирует новую эвристику has_high_cardinality_categoricals."""
    high_cardinality_data = [f"category_{i}" for i in range(101)]
    
    df = pd.DataFrame({
        "id": list(range(101)),
        "high_card_col": high_cardinality_data,  
        "low_card_col": (["A", "B", "C"] * 34)[:101],  
        "numeric_col": list(range(101))
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    assert flags["has_high_cardinality_categoricals"] is True
    
    assert len(flags["high_cardinality_columns"]) > 0
    
    for col, count in flags["high_cardinality_columns"]:
        if col == "high_card_col":
            assert count == 101
            break
    else:
        assert False, "high_card_col не найден в списке high_cardinality_columns"


def test_compute_quality_flags_no_constant_columns():
    """Тестирует случай когда константных колонок нет."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "col1": ["A", "B", "A", "B"],
        "col2": [1.0, 2.0, 3.0, 4.0]
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    
    assert flags["has_constant_columns"] is False
    assert len(flags["constant_columns"]) == 0


def test_compute_quality_flags_no_high_cardinality():
    """Тестирует случай когда нет категориальных колонок с высокой кардинальностью."""
    df = pd.DataFrame({
        "id": list(range(50)),
        "category": (["A", "B", "C"] * 17)[:50],  
        "value": list(range(50))
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    
    assert flags["has_high_cardinality_categoricals"] is False
    assert len(flags["high_cardinality_columns"]) == 0