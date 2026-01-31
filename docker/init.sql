-- Database initialization script
-- This runs when PostgreSQL container first starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE etf_assistant TO etf_user;

-- Create schema if needed
CREATE SCHEMA IF NOT EXISTS public;

-- Note: Tables will be created by Alembic migrations
-- This file ensures database is ready for migrations
