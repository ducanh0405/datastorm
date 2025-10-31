-- ===================================================
-- ==  SƠ ĐỒ THỰC THI (EXECUTION PLAN) WORKSTREAM 1  ==
-- ==  (Optimized SQL Schema - Olist)               ==
-- ===================================================

-- --- BẢNG TRUNG TÂM (CORE ENTITIES) ---

CREATE TABLE customers ( -- Define parent tables first
    customer_id VARCHAR(32) PRIMARY KEY NOT NULL,
    customer_unique_id VARCHAR(32) NOT NULL,
    customer_zip_code_prefix CHAR(5), -- Fixed length
    customer_city VARCHAR(100),
    customer_state CHAR(2) -- Fixed length (e.g., 'SP')
);

CREATE TABLE sellers (
    seller_id VARCHAR(32) PRIMARY KEY NOT NULL,
    seller_zip_code_prefix CHAR(5),
    seller_city VARCHAR(100),
    seller_state CHAR(2)
);

CREATE TABLE products (
    product_id VARCHAR(32) PRIMARY KEY NOT NULL,
    product_category_name VARCHAR(100),
    product_weight_g INT -- Weight is likely integer grams
);

CREATE TABLE orders (
    order_id VARCHAR(32) PRIMARY KEY NOT NULL,
    customer_id VARCHAR(32) NOT NULL,
    order_status VARCHAR(20) NOT NULL, -- Added NOT NULL
    order_purchase_timestamp DATETIME2, -- Use DATETIME2 for precision
    order_delivered_customer_date DATETIME2,
    order_estimated_delivery_date DATETIME2,
    CONSTRAINT fk_orders_customers FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
    order_id VARCHAR(32) NOT NULL,
    order_item_id INT NOT NULL,
    product_id VARCHAR(32), -- Allow NULL if product info is missing? Check data. Assume NOT NULL for now.
    seller_id VARCHAR(32) NOT NULL,
    price DECIMAL(10, 2), -- Use DECIMAL for currency
    freight_value DECIMAL(10, 2), -- Use DECIMAL for currency
    PRIMARY KEY (order_id, order_item_id), -- Composite PK
    CONSTRAINT fk_items_orders FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT fk_items_products FOREIGN KEY (product_id) REFERENCES products(product_id),
    CONSTRAINT fk_items_sellers FOREIGN KEY (seller_id) REFERENCES sellers(seller_id)
);

-- --- BẢNG CẦN XỬ LÝ ĐẶC BIỆT (SPECIAL HANDLING) ---

CREATE TABLE order_reviews (
    review_id VARCHAR(32) NOT NULL, -- May not be unique across orders, composite key better
    order_id VARCHAR(32) NOT NULL,
    review_score TINYINT, -- Score is likely 1-5
    -- review_creation_date DATETIME2, -- Consider adding relevant review dates if needed
    -- review_answer_timestamp DATETIME2,
    PRIMARY KEY (order_id, review_id), -- Composite key is safer
    CONSTRAINT fk_reviews_orders FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

-- Bảng payments GỐC (RAW) - Nguồn của "Bẫy Hợp nhất" 💣
CREATE TABLE order_payments_raw (
    order_id VARCHAR(32) NOT NULL,
    payment_sequential INT,
    payment_type VARCHAR(20),
    payment_installments TINYINT, -- Installments usually small integer
    payment_value DECIMAL(10, 2),
    CONSTRAINT fk_payments_raw_orders FOREIGN KEY (order_id) REFERENCES orders(order_id)
    /*
    *** CẢNH BÁO: BẪY HỢP NHẤT (MERGE TRAP) 💣 ***
    Quan hệ 1-Nhiều với 'orders'. KHÔNG JOIN TRỰC TIẾP VỚI 'order_items'.
    -> YÊU CẦU: Aggregate (GROUP BY order_id) TRƯỚC KHI JOIN.
       Xem bảng 'order_payments_agg'.
    */
);

-- Bảng payments ĐÃ GỘP (AGGREGATED) - An toàn để join ✅
CREATE TABLE order_payments_agg (
    order_id VARCHAR(32) PRIMARY KEY NOT NULL, -- PK (sau khi aggregate)
    payment_installments_total INT, -- Sum can exceed TINYINT
    payment_value_total DECIMAL(10, 2),
    payment_type_primary VARCHAR(20), -- Store the main payment type
    CONSTRAINT fk_payments_agg_orders FOREIGN KEY (order_id) REFERENCES orders(order_id)
    /*
    NOTE: Bảng này được tạo từ 'order_payments_raw' (GROUP BY order_id).
    An toàn (1-1) để JOIN với 'orders'.
    */
);


-- --- BẢNG ƯU TIÊN THẤP (LOW PRIORITY - V2) ---

CREATE TABLE geolocation (
    -- No simple PK, zip code repeats. Query using zip code prefix.
    geolocation_zip_code_prefix CHAR(5),
    geolocation_lat DECIMAL(10, 8), -- More precision for lat/lng
    geolocation_lng DECIMAL(11, 8), -- More precision for lat/lng
    geolocation_city VARCHAR(100),
    geolocation_state CHAR(2)
    /*
    *** ƯU TIÊN THẤP (PoC V1) ✂️ ***
    Rất lớn (~1M rows). Chi phí JOIN cao.
    -> Kế hoạch V1: Dùng `customer_state`/`seller_state` làm proxy.
    -> Kế hoạch V2: Aggregate bảng này theo 'zip_code_prefix' trước khi JOIN.
    */
);

-- Optional: Add Indexes for performance on FK columns and commonly filtered columns
-- CREATE INDEX idx_orders_customer_id ON orders(customer_id);
-- CREATE INDEX idx_items_product_id ON order_items(product_id);
-- CREATE INDEX idx_items_seller_id ON order_items(seller_id);
-- CREATE INDEX idx_geo_zip ON geolocation(geolocation_zip_code_prefix);

PRINT 'Schema created successfully (dropped existing tables if they existed).';