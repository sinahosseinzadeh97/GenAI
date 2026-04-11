from sqlalchemy.orm import Session
from querymind.database.history import QueryHistory
import time

async def save_query(
    db: Session,
    session_id: str,
    user_question: str,
    sql_generated: str | None,
    result_data: list[dict] | None,
    execution_time_ms: int,
    status: str = "success",
    error_message: str | None = None
):
    entry = QueryHistory(
        session_id=session_id,
        user_question=user_question,
        sql_generated=sql_generated,
        result_data=result_data,
        result_row_count=len(result_data) if result_data else 0,
        execution_time_ms=execution_time_ms,
        status=status,
        error_message=error_message
    )
    db.add(entry)
    db.commit()
    return entry

async def get_history(
    session_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
    status: str | None = None
) -> list[dict]:
    from querymind.database.history import engine
    with Session(engine) as db:
        query = db.query(QueryHistory)
        if session_id:
            query = query.filter(QueryHistory.session_id == session_id)
        if status:
            query = query.filter(QueryHistory.status == status)
        results = query.order_by(QueryHistory.created_at.desc())\
                       .offset(offset).limit(limit).all()
        
        output = []
        for r in results:
            data = r.__dict__.copy()
            data.pop("_sa_instance_state", None)
            output.append(data)
        return output
