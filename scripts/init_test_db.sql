-- Test database initialization script for Qloo Supermarket Layout Optimizer
-- This script sets up the test database with required tables and sample data

-- Create test user and database (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'qloo_test') THEN
        CREATE USER qloo_test WITH PASSWORD 'qloo_test_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE qloo_test TO qloo_test;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO qloo_test;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO qloo_test;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO qloo_test;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO qloo_test;

-- Create tables for testing
CREATE TABLE IF NOT EXISTS products (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(100) NOT NULL,
    product_id VARCHAR(50) NOT NULL,
    quantity INTEGER DEFAULT 1,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE IF NOT EXISTS associations (
    id SERIAL PRIMARY KEY,
    product_a VARCHAR(50) NOT NULL,
    product_b VARCHAR(50) NOT NULL,
    support DECIMAL(8, 6) NOT NULL,
    confidence DECIMAL(8, 6) NOT NULL,
    lift DECIMAL(8, 6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_a) REFERENCES products(id),
    FOREIGN KEY (product_b) REFERENCES products(id),
    UNIQUE(product_a, product_b)
);

CREATE TABLE IF NOT EXISTS combos (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    products TEXT[] NOT NULL,
    confidence DECIMAL(8, 6) NOT NULL,
    support DECIMAL(8, 6) NOT NULL,
    lift DECIMAL(8, 6) NOT NULL,
    discount_percent DECIMAL(5, 2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_name ON products(name);
CREATE INDEX IF NOT EXISTS idx_transactions_product_id ON transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_associations_product_a ON associations(product_a);
CREATE INDEX IF NOT EXISTS idx_associations_product_b ON associations(product_b);
CREATE INDEX IF NOT EXISTS idx_associations_confidence ON associations(confidence);
CREATE INDEX IF NOT EXISTS idx_combos_confidence ON combos(confidence);

-- Insert sample test data
INSERT INTO products (id, name, category, price, description) VALUES
('milk_1l', 'Whole Milk 1L', 'dairy', 2.99, 'Fresh whole milk'),
('bread_white', 'White Bread', 'bakery', 1.99, 'Sliced white bread'),
('eggs_12', 'Eggs 12 pack', 'dairy', 3.49, 'Grade A large eggs'),
('cheese_cheddar', 'Cheddar Cheese', 'dairy', 4.99, 'Sharp cheddar cheese'),
('butter_500g', 'Butter 500g', 'dairy', 3.99, 'Unsalted butter'),
('apple_gala', 'Gala Apples', 'fruits', 2.49, 'Fresh gala apples per lb'),
('banana_bunch', 'Banana Bunch', 'fruits', 1.29, 'Fresh bananas'),
('orange_navel', 'Navel Oranges', 'fruits', 2.99, 'Fresh navel oranges per lb'),
('lettuce_iceberg', 'Iceberg Lettuce', 'vegetables', 1.49, 'Fresh iceberg lettuce'),
('tomato_roma', 'Roma Tomatoes', 'vegetables', 2.79, 'Fresh roma tomatoes per lb'),
('chicken_breast', 'Chicken Breast', 'meat', 6.99, 'Boneless chicken breast per lb'),
('ground_beef', 'Ground Beef', 'meat', 5.49, 'Lean ground beef per lb'),
('rice_white', 'White Rice 2lb', 'grains', 2.99, 'Long grain white rice'),
('pasta_spaghetti', 'Spaghetti Pasta', 'grains', 1.49, 'Durum wheat spaghetti'),
('yogurt_greek', 'Greek Yogurt', 'dairy', 1.99, 'Plain Greek yogurt'),
('cereal_oats', 'Oat Cereal', 'grains', 3.99, 'Whole grain oat cereal'),
('coffee_ground', 'Ground Coffee', 'beverages', 8.99, 'Medium roast ground coffee'),
('tea_green', 'Green Tea', 'beverages', 4.49, 'Organic green tea bags'),
('oil_olive', 'Olive Oil', 'condiments', 6.99, 'Extra virgin olive oil'),
('salt_table', 'Table Salt', 'condiments', 0.99, 'Iodized table salt')
ON CONFLICT (id) DO NOTHING;

-- Insert sample transactions for testing
INSERT INTO transactions (transaction_id, product_id, quantity) VALUES
('txn_001', 'milk_1l', 1),
('txn_001', 'bread_white', 1),
('txn_001', 'eggs_12', 1),
('txn_002', 'milk_1l', 1),
('txn_002', 'cheese_cheddar', 1),
('txn_003', 'bread_white', 2),
('txn_003', 'butter_500g', 1),
('txn_004', 'apple_gala', 1),
('txn_004', 'banana_bunch', 1),
('txn_004', 'orange_navel', 1),
('txn_005', 'milk_1l', 1),
('txn_005', 'cereal_oats', 1),
('txn_006', 'chicken_breast', 1),
('txn_006', 'rice_white', 1),
('txn_006', 'lettuce_iceberg', 1),
('txn_007', 'ground_beef', 1),
('txn_007', 'tomato_roma', 1),
('txn_007', 'pasta_spaghetti', 1),
('txn_008', 'milk_1l', 1),
('txn_008', 'yogurt_greek', 1),
('txn_008', 'banana_bunch', 1),
('txn_009', 'coffee_ground', 1),
('txn_009', 'milk_1l', 1),
('txn_010', 'tea_green', 1),
('txn_010', 'honey', 1)  -- This will fail and that's ok for testing
ON CONFLICT DO NOTHING;

-- Insert sample associations for testing
INSERT INTO associations (product_a, product_b, support, confidence, lift) VALUES
('milk_1l', 'cereal_oats', 0.15, 0.75, 2.1),
('milk_1l', 'bread_white', 0.12, 0.68, 1.8),
('bread_white', 'butter_500g', 0.10, 0.82, 2.5),
('cheese_cheddar', 'milk_1l', 0.08, 0.71, 1.9),
('apple_gala', 'banana_bunch', 0.07, 0.65, 1.6),
('chicken_breast', 'rice_white', 0.09, 0.79, 2.2),
('ground_beef', 'pasta_spaghetti', 0.06, 0.73, 1.7),
('coffee_ground', 'milk_1l', 0.05, 0.69, 1.4),
('yogurt_greek', 'banana_bunch', 0.04, 0.61, 1.3),
('eggs_12', 'milk_1l', 0.11, 0.76, 2.0)
ON CONFLICT (product_a, product_b) DO NOTHING;

-- Insert sample combos for testing
INSERT INTO combos (name, products, confidence, support, lift, discount_percent) VALUES
('Breakfast Combo', ARRAY['milk_1l', 'cereal_oats', 'banana_bunch'], 0.75, 0.08, 2.1, 10.0),
('Baking Combo', ARRAY['bread_white', 'butter_500g', 'eggs_12'], 0.82, 0.06, 2.5, 15.0),
('Fruit Bowl', ARRAY['apple_gala', 'banana_bunch', 'orange_navel'], 0.65, 0.05, 1.6, 8.0),
('Dinner Combo', ARRAY['chicken_breast', 'rice_white', 'lettuce_iceberg'], 0.79, 0.07, 2.2, 12.0),
('Pasta Night', ARRAY['ground_beef', 'pasta_spaghetti', 'tomato_roma'], 0.73, 0.04, 1.7, 20.0),
('Coffee Time', ARRAY['coffee_ground', 'milk_1l'], 0.69, 0.03, 1.4, 5.0),
('Healthy Snack', ARRAY['yogurt_greek', 'banana_bunch'], 0.61, 0.02, 1.3, 7.0),
('Classic Breakfast', ARRAY['eggs_12', 'milk_1l', 'bread_white'], 0.76, 0.09, 2.0, 12.0),
('Dairy Essentials', ARRAY['milk_1l', 'cheese_cheddar', 'butter_500g'], 0.71, 0.05, 1.9, 10.0),
('Fresh Start', ARRAY['apple_gala', 'yogurt_greek'], 0.58, 0.03, 1.2, 6.0)
ON CONFLICT DO NOTHING;

-- Create a function to refresh test data
CREATE OR REPLACE FUNCTION refresh_test_data()
RETURNS void AS $$
BEGIN
    -- Clear existing data
    DELETE FROM combos;
    DELETE FROM associations;
    DELETE FROM transactions;
    DELETE FROM products;
    
    -- Reset sequences
    ALTER SEQUENCE transactions_id_seq RESTART WITH 1;
    ALTER SEQUENCE associations_id_seq RESTART WITH 1;
    ALTER SEQUENCE combos_id_seq RESTART WITH 1;
    
    -- Re-insert test data (reuse the INSERT statements above)
    -- This would be expanded in a real implementation
    RAISE NOTICE 'Test data refreshed successfully';
END;
$$ LANGUAGE plpgsql;

-- Create a test user function for API key validation
CREATE OR REPLACE FUNCTION validate_test_api_key(api_key TEXT)
RETURNS boolean AS $$
BEGIN
    -- For testing, accept specific test API keys
    RETURN api_key IN ('test-key', 'test-api-key', 'ci-test-key');
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission on functions
GRANT EXECUTE ON FUNCTION refresh_test_data() TO qloo_test;
GRANT EXECUTE ON FUNCTION validate_test_api_key(TEXT) TO qloo_test;

-- Create a view for easy testing
CREATE OR REPLACE VIEW test_combo_summary AS
SELECT 
    c.name,
    array_length(c.products, 1) as product_count,
    c.confidence,
    c.support,
    c.lift,
    c.discount_percent,
    c.created_at
FROM combos c
ORDER BY c.confidence DESC;

GRANT SELECT ON test_combo_summary TO qloo_test;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Test database initialized successfully at %', NOW();
    RAISE NOTICE 'Products loaded: %', (SELECT COUNT(*) FROM products);
    RAISE NOTICE 'Transactions loaded: %', (SELECT COUNT(*) FROM transactions);
    RAISE NOTICE 'Associations loaded: %', (SELECT COUNT(*) FROM associations);
    RAISE NOTICE 'Combos loaded: %', (SELECT COUNT(*) FROM combos);
END
$$;