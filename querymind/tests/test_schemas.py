import pytest
from pydantic import ValidationError

from querymind.schemas.models import QueryRequest, QueryResult, QueryError, ColumnInfo, SQLQuery

def test_query_request_validation():
    # Valid
    req = QueryRequest(nl_query="How many users?")
    assert req.nl_query == "How many users?"
    assert req.max_rows == 50

    # Invalid (missing nl_query)
    with pytest.raises(ValidationError):
        QueryRequest(max_rows=10)

def test_query_result_validation():
    col = ColumnInfo(name="id", type="INTEGER", nullable=True)
    
    # Valid
    res = QueryResult(
        columns=[col],
        rows=[{"id": 1}],
        row_count=1
    )
    assert res.row_count == 1
    
    # Invalid (missing rows)
    with pytest.raises(ValidationError):
        QueryResult(columns=[col], row_count=1)

def test_query_error_validation():
    # Valid
    err = QueryError(code="TEST", message="Test error")
    assert err.detail is None
    
    # Invalid
    with pytest.raises(ValidationError):
        QueryError(message="No code")
