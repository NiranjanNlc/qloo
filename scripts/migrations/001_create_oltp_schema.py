#!/usr/bin/env python3
"""
Migration 001: Create OLTP Schema for POS Data Ingestion

This migration creates the complete OLTP schema optimized for high-volume
point-of-sale transaction data with partitioning and analytics views.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from database_setup import get_database_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Migration001:
    """Migration to create OLTP schema for POS data."""
    
    def __init__(self, database_url=None):
        """Initialize migration with database connection."""
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.migration_name = "001_create_oltp_schema"
        self.migration_version = "1.0.0"
        
    def get_connection(self):
        """Get database connection."""
        if self.database_url:
            return psycopg2.connect(self.database_url)
        else:
            # Use local configuration
            config = get_database_config()
            return psycopg2.connect(
                host=config.get('host', 'localhost'),
                port=config.get('port', 5432),
                database=config.get('database', 'qloo_optimizer'),
                user=config.get('user', 'qloo_user'),
                password=config.get('password', 'qloo_password')
            )
    
    def load_sql_file(self, filename):
        """Load SQL commands from file."""
        sql_path = Path(__file__).parent.parent / filename
        try:
            with open(sql_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"SQL file not found: {sql_path}")
            raise
    
    def execute_sql_commands(self, conn, sql_content):
        """Execute SQL commands with proper transaction management."""
        cursor = conn.cursor()
        
        # Split SQL content into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
        
        for i, statement in enumerate(statements):
            try:
                logger.info(f"Executing statement {i+1}/{len(statements)}")
                cursor.execute(statement)
                conn.commit()
                logger.debug(f"Statement executed successfully: {statement[:100]}...")
                
            except Exception as e:
                logger.error(f"Error executing statement {i+1}: {e}")
                logger.error(f"Statement: {statement[:200]}...")
                conn.rollback()
                raise
        
        cursor.close()
    
    def create_migration_tracking(self, conn):
        """Create migration tracking table if it doesn't exist."""
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration_name VARCHAR(100) UNIQUE NOT NULL,
                migration_version VARCHAR(20) NOT NULL,
                executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                execution_time_seconds DECIMAL(10,3),
                status VARCHAR(20) DEFAULT 'success',
                error_message TEXT,
                rollback_sql TEXT
            )
        """)
        conn.commit()
        cursor.close()
    
    def is_migration_applied(self, conn):
        """Check if migration has already been applied."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM schema_migrations WHERE migration_name = %s AND status = 'success'",
            (self.migration_name,)
        )
        count = cursor.fetchone()[0]
        cursor.close()
        return count > 0
    
    def record_migration(self, conn, execution_time, status='success', error_message=None):
        """Record migration execution in tracking table."""
        cursor = conn.cursor()
        
        rollback_sql = self.generate_rollback_sql()
        
        cursor.execute("""
            INSERT INTO schema_migrations 
            (migration_name, migration_version, execution_time_seconds, status, error_message, rollback_sql)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (migration_name) DO UPDATE SET
                executed_at = CURRENT_TIMESTAMP,
                execution_time_seconds = EXCLUDED.execution_time_seconds,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                rollback_sql = EXCLUDED.rollback_sql
        """, (
            self.migration_name,
            self.migration_version,
            execution_time,
            status,
            error_message,
            rollback_sql
        ))
        
        conn.commit()
        cursor.close()
    
    def generate_rollback_sql(self):
        """Generate SQL commands to rollback this migration."""
        return """
        -- Rollback for 001_create_oltp_schema
        DROP SCHEMA IF EXISTS pos_oltp CASCADE;
        DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
        DROP FUNCTION IF EXISTS create_partition_indexes(TEXT, TEXT) CASCADE;
        DROP FUNCTION IF EXISTS create_monthly_partitions(INTEGER) CASCADE;
        DROP FUNCTION IF EXISTS drop_old_partitions(INTEGER) CASCADE;
        DROP ROLE IF EXISTS pos_readonly;
        DROP ROLE IF EXISTS pos_write;
        DROP ROLE IF EXISTS pos_admin;
        """
    
    def validate_schema(self, conn):
        """Validate that the schema was created correctly."""
        cursor = conn.cursor()
        
        # Check if schema exists
        cursor.execute("SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'pos_oltp')")
        if not cursor.fetchone()[0]:
            raise Exception("pos_oltp schema was not created")
        
        # Check critical tables
        critical_tables = ['stores', 'categories', 'products', 'transactions', 'transaction_items']
        for table in critical_tables:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'pos_oltp' AND table_name = %s
                )
            """, (table,))
            if not cursor.fetchone()[0]:
                raise Exception(f"Critical table {table} was not created")
        
        # Check partitions were created
        cursor.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'pos_oltp' 
            AND (table_name LIKE 'transactions_%' OR table_name LIKE 'transaction_items_%')
        """)
        partition_count = cursor.fetchone()[0]
        if partition_count < 4:  # Should have at least a few partitions
            logger.warning(f"Only {partition_count} partitions found, expected more")
        
        # Check views were created
        views = ['current_product_placements', 'daily_transaction_summary', 'product_performance']
        for view in views:
            cursor.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.views 
                    WHERE table_schema = 'pos_oltp' AND table_name = %s
                )
            """, (view,))
            if not cursor.fetchone()[0]:
                raise Exception(f"View {view} was not created")
        
        cursor.close()
        logger.info("Schema validation passed")
    
    def up(self):
        """Apply the migration."""
        logger.info(f"Starting migration: {self.migration_name}")
        start_time = datetime.now()
        
        try:
            # Get database connection
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Create migration tracking table
            self.create_migration_tracking(conn)
            
            # Check if migration already applied
            if self.is_migration_applied(conn):
                logger.info("Migration already applied, skipping")
                conn.close()
                return
            
            # Load and execute schema SQL
            schema_sql = self.load_sql_file('oltp_schema.sql')
            self.execute_sql_commands(conn, schema_sql)
            
            # Validate schema creation
            self.validate_schema(conn)
            
            # Record successful migration
            execution_time = (datetime.now() - start_time).total_seconds()
            self.record_migration(conn, execution_time)
            
            conn.close()
            logger.info(f"Migration {self.migration_name} completed successfully in {execution_time:.2f} seconds")
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            logger.error(f"Migration failed: {error_msg}")
            
            try:
                # Record failed migration
                conn = self.get_connection()
                self.create_migration_tracking(conn)
                self.record_migration(conn, execution_time, 'failed', error_msg)
                conn.close()
            except:
                logger.error("Failed to record migration failure")
            
            raise
    
    def down(self):
        """Rollback the migration."""
        logger.info(f"Rolling back migration: {self.migration_name}")
        
        try:
            conn = self.get_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            # Execute rollback SQL
            rollback_sql = self.generate_rollback_sql()
            self.execute_sql_commands(conn, rollback_sql)
            
            # Remove from migration tracking
            cursor = conn.cursor()
            cursor.execute("DELETE FROM schema_migrations WHERE migration_name = %s", (self.migration_name,))
            conn.commit()
            cursor.close()
            
            conn.close()
            logger.info(f"Migration {self.migration_name} rolled back successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

def main():
    """Main function to run migration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run OLTP schema migration')
    parser.add_argument('--action', choices=['up', 'down'], default='up',
                      help='Migration action: up (apply) or down (rollback)')
    parser.add_argument('--database-url', help='Database connection URL')
    parser.add_argument('--dry-run', action='store_true', 
                      help='Show what would be executed without running')
    
    args = parser.parse_args()
    
    migration = Migration001(args.database_url)
    
    if args.dry_run:
        print(f"Would execute migration {migration.migration_name} with action: {args.action}")
        if args.action == 'down':
            print("Rollback SQL:")
            print(migration.generate_rollback_sql())
        return
    
    try:
        if args.action == 'up':
            migration.up()
        else:
            migration.down()
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 