import csv
from typing import Literal
from datetime import datetime

from querymind.schemas.models import ImportResult
from querymind.database.engine import get_db_connection

def _infer_type(value: str) -> str:
    """Infer the SQLite data type for a string value."""
    if not value or value.isspace():
        return "TEXT"
    
    # Try integer
    try:
        int(value)
        return "INTEGER"
    except ValueError:
        pass
        
    # Try float (REAL)
    try:
        float(value)
        return "REAL"
    except ValueError:
        pass
        
    # Try datetime
    try:
        if len(value) >= 10:
            # simple check for iso format
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return "DATETIME"
    except ValueError:
        pass
        
    return "TEXT"


async def import_csv_to_table(
    csv_path: str,
    table_name: str,
    if_exists: Literal["replace", "append", "fail"] = "replace"
) -> ImportResult:
    """Import a CSV file into a SQLite database table dynamically."""
    
    # Read the file
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
            
        columns = [str(x).strip() for x in reader.fieldnames]
        
        # Read all rows into memory
        rows = list(reader)

    if not rows:
        return ImportResult(table_name=table_name, rows_imported=0, columns=columns)

    # Infer column types from the first data row
    first_row = rows[0]
    column_types = {}
    for col in columns:
        val = str(first_row.get(col, "")).strip()
        column_types[col] = _infer_type(val)

    async with get_db_connection() as db:
        if if_exists == "replace":
            await db.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            
        # Check if table exists
        async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)) as cursor:
            table_exists = await cursor.fetchone() is not None
            
        if table_exists and if_exists == "fail":
            raise ValueError(f"Table '{table_name}' already exists and if_exists='fail'.")

        if not table_exists:
            # Create table
            cols_def = ", ".join([f'"{col}" {column_types[col]}' for col in columns])
            create_sql = f'CREATE TABLE "{table_name}" ({cols_def})'
            await db.execute(create_sql)
            
        # Insert rows in batches of 500
        cols_csv = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f'INSERT INTO "{table_name}" ({cols_csv}) VALUES ({placeholders})'
        
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            batch_data = []
            for row in batch:
                batch_data.append(tuple(row.get(col) for col in columns))
                
            await db.executemany(insert_sql, batch_data)
            
        await db.commit()
            
    return ImportResult(
        table_name=table_name,
        rows_imported=len(rows),
        columns=[f"{c} ({column_types[c]})" for c in columns]
    )
