import argparse
import asyncio
import sys

def main():
    parser = argparse.ArgumentParser(description="QueryMind CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # seed
    subparsers.add_parser("seed", help="Seed e-commerce demo data")
    
    # import-csv
    import_parser = subparsers.add_parser("import-csv", help="Import a CSV file to a table")
    import_parser.add_argument("path", help="Path to the CSV file")
    import_parser.add_argument("table_name", help="Name of the table to create/insert")
    
    # tables
    subparsers.add_parser("tables", help="List all tables and row counts")
    
    # server
    subparsers.add_parser("server", help="Start MCP server")
    
    # api
    subparsers.add_parser("api", help="Start FastAPI server")
    
    args = parser.parse_args()
    
    if args.command == "seed":
        from querymind.database.seed import seed_database as seed_main
        asyncio.run(seed_main())
    elif args.command == "import-csv":
        from querymind.tools.import_tool import import_csv_to_table
        try:
            result = asyncio.run(import_csv_to_table(args.path, args.table_name))
            print(f"Imported {result.rows_imported} rows into '{result.table_name}'.")
            print("Columns detected:")
            for col in result.columns:
                print(f"  - {col}")
        except Exception as e:
            print(f"Import failed: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.command == "tables":
        from querymind.database.router import execute_query
        
        async def list_tables():
            # Get table names
            result = await execute_query("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r['name'] for r in result.rows if r['name'] != 'sqlite_sequence']
            
            if not tables:
                print("No tables found in database.")
                return
                
            print("Tables:")
            for t in tables:
                cnt = await execute_query(f'SELECT count(*) as cnt FROM "{t}"')
                row_count = cnt.rows[0]['cnt']
                print(f"  - {t}: {row_count} rows")
                
        asyncio.run(list_tables())
    elif args.command == "server":
        from querymind.server import main as server_main
        asyncio.run(server_main())
    elif args.command == "api":
        import uvicorn
        uvicorn.run(
            "querymind.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False
        )

if __name__ == "__main__":
    sys.exit(main())
