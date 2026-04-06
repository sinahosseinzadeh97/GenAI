"""
OpenAI function-calling tool schemas for the Contract Agent.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_contracts",
            "description": "Searches the document base for relevant contract information matching the query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "The number of top results to return. Default is 5."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_field",
            "description": "Extracts a specific field (like 'Total Cost' or 'Effective Date') from the documents based on a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query identifying which contract or context to extract from."
                    },
                    "field": {
                        "type": "string",
                        "description": "The specific field to extract from the document."
                    }
                },
                "required": ["query", "field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_contracts",
            "description": "Compares multiple contracts against a query to find similarities and differences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The comparison query describing what aspects to compare."
                    },
                    "filenames": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "A list of specific filenames to compare."
                    }
                },
                "required": ["query", "filenames"]
            }
        }
    }
]
