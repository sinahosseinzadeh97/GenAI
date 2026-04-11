-- Create read-only user for SQL query execution
CREATE USER querymind_read WITH PASSWORD 'querymind_read_pass';

-- Grant connection
GRANT CONNECT ON DATABASE querymind TO querymind_read;

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO querymind_read;

-- Grant SELECT only on all current tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO querymind_read;

-- Grant SELECT on all future tables automatically
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO querymind_read;
