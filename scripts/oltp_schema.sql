-- OLTP Schema for POS Data Ingestion
-- Designed for high-volume transactional data with optimized indexing
-- Supports real-time analytics and association rule mining

-- Create schema
CREATE SCHEMA IF NOT EXISTS pos_oltp;

-- Set search path
SET search_path TO pos_oltp, public;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ===============================================
-- DIMENSION TABLES
-- ===============================================

-- Stores table
CREATE TABLE stores (
    store_id VARCHAR(20) PRIMARY KEY,
    store_name VARCHAR(100) NOT NULL,
    store_type VARCHAR(50) DEFAULT 'supermarket',
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    timezone VARCHAR(50) DEFAULT 'UTC',
    opening_hours JSONB,
    square_footage INTEGER,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Categories table (hierarchical)
CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_code VARCHAR(20) UNIQUE NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    parent_category_id INTEGER REFERENCES categories(category_id),
    category_level INTEGER DEFAULT 1,
    category_path VARCHAR(500), -- For hierarchical queries
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    product_id VARCHAR(50) PRIMARY KEY,
    sku VARCHAR(50) UNIQUE NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    brand VARCHAR(100),
    category_id INTEGER NOT NULL REFERENCES categories(category_id),
    subcategory VARCHAR(100),
    unit_of_measure VARCHAR(20) DEFAULT 'EACH',
    pack_size DECIMAL(10,3),
    weight_kg DECIMAL(10,3),
    dimensions JSONB, -- {length, width, height}
    barcode VARCHAR(50),
    supplier_id VARCHAR(50),
    cost_price DECIMAL(10,2),
    list_price DECIMAL(10,2) NOT NULL,
    is_perishable BOOLEAN DEFAULT FALSE,
    shelf_life_days INTEGER,
    storage_temperature VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Store sections/departments
CREATE TABLE store_sections (
    section_id SERIAL PRIMARY KEY,
    store_id VARCHAR(20) NOT NULL REFERENCES stores(store_id),
    section_code VARCHAR(20) NOT NULL,
    section_name VARCHAR(100) NOT NULL,
    section_type VARCHAR(50), -- 'department', 'aisle', 'shelf'
    parent_section_id INTEGER REFERENCES store_sections(section_id),
    location_coordinates POINT, -- (x, y) coordinates in store
    capacity INTEGER,
    temperature_zone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(store_id, section_code)
);

-- Product placements in store sections
CREATE TABLE product_placements (
    placement_id SERIAL PRIMARY KEY,
    store_id VARCHAR(20) NOT NULL REFERENCES stores(store_id),
    product_id VARCHAR(50) NOT NULL REFERENCES products(product_id),
    section_id INTEGER NOT NULL REFERENCES store_sections(section_id),
    shelf_position VARCHAR(20),
    facing_count INTEGER DEFAULT 1,
    is_primary_location BOOLEAN DEFAULT TRUE,
    start_date DATE NOT NULL,
    end_date DATE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(store_id, product_id, section_id, start_date)
);

-- Customers table (optional, for loyalty programs)
CREATE TABLE customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_type VARCHAR(20) DEFAULT 'regular', -- 'loyalty', 'employee', 'guest'
    registration_date DATE,
    demographics JSONB,
    preferences JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ===============================================
-- FACT TABLES (High Volume)
-- ===============================================

-- Main transactions table
CREATE TABLE transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    store_id VARCHAR(20) NOT NULL REFERENCES stores(store_id),
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    cashier_id VARCHAR(50),
    register_id VARCHAR(20),
    transaction_date DATE NOT NULL,
    transaction_time TIME NOT NULL,
    transaction_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    discount_amount DECIMAL(10,2) DEFAULT 0,
    payment_method VARCHAR(20), -- 'cash', 'card', 'mobile', 'mixed'
    payment_details JSONB,
    item_count INTEGER NOT NULL,
    is_return BOOLEAN DEFAULT FALSE,
    original_transaction_id VARCHAR(50),
    session_id VARCHAR(100), -- For basket analysis
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (transaction_date);

-- Transaction items table
CREATE TABLE transaction_items (
    item_id BIGSERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    product_id VARCHAR(50) NOT NULL REFERENCES products(product_id),
    quantity DECIMAL(10,3) NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL,
    discount_percent DECIMAL(5,2) DEFAULT 0,
    discount_amount DECIMAL(10,2) DEFAULT 0,
    promotion_id VARCHAR(50),
    section_id INTEGER REFERENCES store_sections(section_id),
    scan_timestamp TIMESTAMP WITH TIME ZONE,
    is_return BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (created_at);

-- ===============================================
-- PARTITIONING SETUP
-- ===============================================

-- Create monthly partitions for transactions (last 2 years + 6 months ahead)
DO $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE - INTERVAL '24 months');
    end_date DATE := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '6 months');
    current_month DATE := start_date;
    partition_name TEXT;
    next_month DATE;
BEGIN
    WHILE current_month < end_date LOOP
        next_month := current_month + INTERVAL '1 month';
        partition_name := 'transactions_' || TO_CHAR(current_month, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE %I PARTITION OF transactions
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, current_month, next_month);
            
        current_month := next_month;
    END LOOP;
END $$;

-- Create monthly partitions for transaction_items
DO $$
DECLARE
    start_date TIMESTAMP := DATE_TRUNC('month', CURRENT_TIMESTAMP - INTERVAL '24 months');
    end_date TIMESTAMP := DATE_TRUNC('month', CURRENT_TIMESTAMP + INTERVAL '6 months');
    current_month TIMESTAMP := start_date;
    partition_name TEXT;
    next_month TIMESTAMP;
BEGIN
    WHILE current_month < end_date LOOP
        next_month := current_month + INTERVAL '1 month';
        partition_name := 'transaction_items_' || TO_CHAR(current_month, 'YYYY_MM');
        
        EXECUTE format('
            CREATE TABLE %I PARTITION OF transaction_items
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, current_month, next_month);
            
        current_month := next_month;
    END LOOP;
END $$;

-- ===============================================
-- INDEXES FOR PERFORMANCE
-- ===============================================

-- Stores indexes
CREATE INDEX idx_stores_active ON stores(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_stores_type ON stores(store_type);

-- Categories indexes
CREATE INDEX idx_categories_parent ON categories(parent_category_id);
CREATE INDEX idx_categories_active ON categories(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_categories_path ON categories USING GIN(to_tsvector('english', category_path));

-- Products indexes
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_products_price_range ON products(list_price);
CREATE INDEX idx_products_barcode ON products(barcode);

-- Store sections indexes
CREATE INDEX idx_store_sections_store ON store_sections(store_id);
CREATE INDEX idx_store_sections_parent ON store_sections(parent_section_id);
CREATE INDEX idx_store_sections_location ON store_sections USING GIST(location_coordinates);

-- Product placements indexes
CREATE INDEX idx_product_placements_store_product ON product_placements(store_id, product_id);
CREATE INDEX idx_product_placements_section ON product_placements(section_id);
CREATE INDEX idx_product_placements_dates ON product_placements(start_date, end_date);

-- Transactions indexes (created on each partition)
-- These will be created by the partitioning trigger

-- Transaction items indexes (created on each partition)
-- These will be created by the partitioning trigger

-- ===============================================
-- TRIGGERS AND FUNCTIONS
-- ===============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_stores_updated_at BEFORE UPDATE ON stores
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create indexes on new partitions
CREATE OR REPLACE FUNCTION create_partition_indexes(partition_name TEXT, base_table TEXT)
RETURNS VOID AS $$
BEGIN
    IF base_table = 'transactions' THEN
        EXECUTE format('CREATE INDEX idx_%s_store_date ON %I(store_id, transaction_date)', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_timestamp ON %I(transaction_timestamp)', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_customer ON %I(customer_id) WHERE customer_id IS NOT NULL', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_amount ON %I(total_amount)', partition_name, partition_name);
    ELSIF base_table = 'transaction_items' THEN
        EXECUTE format('CREATE INDEX idx_%s_transaction ON %I(transaction_id)', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_product ON %I(product_id)', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_section ON %I(section_id) WHERE section_id IS NOT NULL', partition_name, partition_name);
        EXECUTE format('CREATE INDEX idx_%s_price ON %I(unit_price)', partition_name, partition_name);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create indexes on existing partitions
DO $$
DECLARE
    partition_record RECORD;
BEGIN
    -- Create indexes for transaction partitions
    FOR partition_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'transactions_%' AND schemaname = 'pos_oltp'
    LOOP
        PERFORM create_partition_indexes(partition_record.tablename, 'transactions');
    END LOOP;
    
    -- Create indexes for transaction_items partitions
    FOR partition_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'transaction_items_%' AND schemaname = 'pos_oltp'
    LOOP
        PERFORM create_partition_indexes(partition_record.tablename, 'transaction_items');
    END LOOP;
END $$;

-- ===============================================
-- VIEWS FOR ANALYTICS
-- ===============================================

-- View for current product placements
CREATE VIEW current_product_placements AS
SELECT 
    pp.*,
    p.product_name,
    p.brand,
    p.category_id,
    c.category_name,
    ss.section_name,
    ss.section_type
FROM product_placements pp
JOIN products p ON pp.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
JOIN store_sections ss ON pp.section_id = ss.section_id
WHERE pp.start_date <= CURRENT_DATE 
  AND (pp.end_date IS NULL OR pp.end_date >= CURRENT_DATE)
  AND pp.is_primary_location = TRUE;

-- View for daily transaction summary
CREATE VIEW daily_transaction_summary AS
SELECT 
    store_id,
    transaction_date,
    COUNT(*) as transaction_count,
    SUM(total_amount) as total_revenue,
    SUM(tax_amount) as total_tax,
    SUM(discount_amount) as total_discounts,
    SUM(item_count) as total_items,
    AVG(total_amount) as avg_basket_size,
    COUNT(DISTINCT customer_id) as unique_customers
FROM transactions
WHERE is_return = FALSE
GROUP BY store_id, transaction_date;

-- View for product performance metrics
CREATE VIEW product_performance AS
SELECT 
    p.product_id,
    p.product_name,
    p.brand,
    c.category_name,
    COUNT(ti.item_id) as times_sold,
    SUM(ti.quantity) as total_quantity_sold,
    SUM(ti.total_price) as total_revenue,
    AVG(ti.unit_price) as avg_selling_price,
    p.cost_price,
    (AVG(ti.unit_price) - p.cost_price) as avg_margin
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN transaction_items ti ON p.product_id = ti.product_id
WHERE p.is_active = TRUE
GROUP BY p.product_id, p.product_name, p.brand, c.category_name, p.cost_price;

-- ===============================================
-- SAMPLE DATA CONSTRAINTS
-- ===============================================

-- Add constraints for data quality
ALTER TABLE transactions ADD CONSTRAINT chk_total_amount_positive CHECK (total_amount >= 0);
ALTER TABLE transactions ADD CONSTRAINT chk_tax_amount_non_negative CHECK (tax_amount >= 0);
ALTER TABLE transactions ADD CONSTRAINT chk_discount_amount_non_negative CHECK (discount_amount >= 0);
ALTER TABLE transactions ADD CONSTRAINT chk_item_count_positive CHECK (item_count > 0);

ALTER TABLE transaction_items ADD CONSTRAINT chk_quantity_positive CHECK (quantity > 0);
ALTER TABLE transaction_items ADD CONSTRAINT chk_unit_price_positive CHECK (unit_price > 0);
ALTER TABLE transaction_items ADD CONSTRAINT chk_total_price_positive CHECK (total_price >= 0);
ALTER TABLE transaction_items ADD CONSTRAINT chk_discount_percent_valid CHECK (discount_percent >= 0 AND discount_percent <= 100);

ALTER TABLE products ADD CONSTRAINT chk_list_price_positive CHECK (list_price > 0);
ALTER TABLE products ADD CONSTRAINT chk_cost_price_non_negative CHECK (cost_price >= 0);

-- ===============================================
-- PERMISSIONS AND SECURITY
-- ===============================================

-- Create roles
CREATE ROLE pos_readonly;
CREATE ROLE pos_write;
CREATE ROLE pos_admin;

-- Grant permissions
GRANT USAGE ON SCHEMA pos_oltp TO pos_readonly, pos_write, pos_admin;
GRANT SELECT ON ALL TABLES IN SCHEMA pos_oltp TO pos_readonly;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pos_oltp TO pos_write;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pos_oltp TO pos_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA pos_oltp TO pos_write, pos_admin;

-- Row Level Security (optional)
-- ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY transactions_store_policy ON transactions FOR ALL TO pos_write USING (store_id = current_setting('app.current_store_id'));

-- ===============================================
-- MAINTENANCE PROCEDURES
-- ===============================================

-- Function to create new monthly partitions
CREATE OR REPLACE FUNCTION create_monthly_partitions(months_ahead INTEGER DEFAULT 3)
RETURNS VOID AS $$
DECLARE
    start_date DATE := DATE_TRUNC('month', CURRENT_DATE);
    end_date DATE := DATE_TRUNC('month', CURRENT_DATE + (months_ahead || ' months')::INTERVAL);
    current_month DATE := start_date;
    partition_name TEXT;
    next_month DATE;
BEGIN
    WHILE current_month < end_date LOOP
        next_month := current_month + INTERVAL '1 month';
        
        -- Create transactions partition
        partition_name := 'transactions_' || TO_CHAR(current_month, 'YYYY_MM');
        
        BEGIN
            EXECUTE format('
                CREATE TABLE %I PARTITION OF transactions
                FOR VALUES FROM (%L) TO (%L)',
                partition_name, current_month, next_month);
            
            PERFORM create_partition_indexes(partition_name, 'transactions');
            RAISE NOTICE 'Created partition: %', partition_name;
        EXCEPTION WHEN duplicate_table THEN
            -- Partition already exists, skip
        END;
        
        -- Create transaction_items partition
        partition_name := 'transaction_items_' || TO_CHAR(current_month, 'YYYY_MM');
        
        BEGIN
            EXECUTE format('
                CREATE TABLE %I PARTITION OF transaction_items
                FOR VALUES FROM (%L) TO (%L)',
                partition_name, current_month::TIMESTAMP, next_month::TIMESTAMP);
            
            PERFORM create_partition_indexes(partition_name, 'transaction_items');
            RAISE NOTICE 'Created partition: %', partition_name;
        EXCEPTION WHEN duplicate_table THEN
            -- Partition already exists, skip
        END;
        
        current_month := next_month;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Function to drop old partitions
CREATE OR REPLACE FUNCTION drop_old_partitions(retention_months INTEGER DEFAULT 24)
RETURNS VOID AS $$
DECLARE
    cutoff_date DATE := DATE_TRUNC('month', CURRENT_DATE - (retention_months || ' months')::INTERVAL);
    partition_record RECORD;
BEGIN
    FOR partition_record IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE (tablename LIKE 'transactions_%' OR tablename LIKE 'transaction_items_%')
          AND schemaname = 'pos_oltp'
          AND RIGHT(tablename, 7) < TO_CHAR(cutoff_date, 'YYYY_MM')
    LOOP
        EXECUTE format('DROP TABLE %I.%I', partition_record.schemaname, partition_record.tablename);
        RAISE NOTICE 'Dropped old partition: %', partition_record.tablename;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Comment the schema
COMMENT ON SCHEMA pos_oltp IS 'OLTP schema optimized for high-volume POS transaction data with partitioning and analytics views';
COMMENT ON TABLE transactions IS 'Main transactions table - partitioned by transaction_date for performance';
COMMENT ON TABLE transaction_items IS 'Transaction line items - partitioned by created_at timestamp';
COMMENT ON TABLE products IS 'Product master data with category hierarchy';
COMMENT ON TABLE product_placements IS 'Historical record of product locations in stores';
COMMENT ON VIEW current_product_placements IS 'Current active product placements across all stores';
COMMENT ON VIEW daily_transaction_summary IS 'Daily aggregated transaction metrics by store';
COMMENT ON VIEW product_performance IS 'Product-level sales and profitability metrics'; 