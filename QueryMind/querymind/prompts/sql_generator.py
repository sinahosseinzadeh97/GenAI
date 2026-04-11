"""
Prompt template for translating natural language into SQL.
"""

SQL_SYSTEM_PROMPT = """You are an expert SQLite database engineer.
Your task is to translate a user's natural language question into a valid, executable SQLite query.

Here is the database schema:
{schema_ddl}

RULES:
1. You MUST output ONLY valid SQLite SQL code.
2. DO NOT include any markdown formatting (e.g., avoid ```sql and ```).
3. DO NOT include explanations, comments, or extra text. Output only the raw SQL.
4. If the question cannot be answered using the provided schema or if tables/columns are missing, return exactly: SELECT 'ERROR: CANNOT_ANSWER' AS error
5. For ambiguous questions, make the most reasonable database assumptions.
6. Use EXACT table and column names as seen in the schema. Do not invent columns.
7. Limit returned data to a maximum of {max_rows} rows using LIMIT (unless it is an aggregation query).
{conversation_history}"""

def get_sql_generation_prompt(
    schema_ddl: str, max_rows: int = 50, conversation_history: str = ""
) -> str:
    """
    Injects the database schema and constraints into the system prompt.
    
    Args:
        schema_ddl: The raw DDL string of the database schema.
        max_rows: The maximum number of rows the prompt should allow.
        
    Returns:
        The formatted system prompt.
    """
    history_str = f"\nPrevious conversation context:\n{conversation_history}" if conversation_history else ""
    return SQL_SYSTEM_PROMPT.format(schema_ddl=schema_ddl, max_rows=max_rows, conversation_history=history_str)

def build_user_prompt(nl_query: str) -> str:
    """
    Constructs the prompt for the user's specific natural language query.
    
    Args:
        nl_query: The natural language string requested by the user.
        
    Returns:
        The text to submit to the LLM representing the user's query.
    """
    return f"Translate the following request into a SQLite query:\n\n{nl_query}"
