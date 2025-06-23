# =============================
# 1. IMPORTING LIBRARIES
# =============================

import pandas as pd
import numpy as np
import duckdb

# ===================================================
# 2. DATA EXPLORATION, CLEANING AND PREPARATION
# ===================================================

# dimension table
product_master = pd.read_parquet('dataset/parquet/PRODUCT_MASTER.parquet')
store_master = pd.read_parquet('dataset/parquet/STORE_MASTER.parquet')
user_master = pd.read_parquet('dataset/parquet/USER_MASTER.parquet')
# fact tables
operation_details = pd.read_parquet('dataset/parquet/FACT_OPERATION_TABLE.parquet')
order_details = pd.read_parquet('dataset/parquet/FACT_ORDER_TABLE.parquet')

print("Product Master Shape:", product_master.shape)
print("Store Master Shape:", store_master.shape)
print("User Master Shape:", user_master.shape)
print("Operation Details Shape:", operation_details.shape)
print("Order Details Shape:", order_details.shape)

print("User Master Info:")
user_master.info()

user_master['REGISTRATION_TIMESTAMP'] = pd.to_datetime(user_master['REGISTRATION_TIMESTAMP'])
user_master['CREATION_DATE'] = pd.to_datetime(user_master['CREATION_DATE'])

print("Order Details Info:")
order_details.info()

order_details['ORDER_TIMESTAMP'] = pd.to_datetime(order_details['ORDER_TIMESTAMP'], errors='coerce')
order_details['ORDER_DATE'] = pd.to_datetime(order_details['ORDER_DATE'], errors='coerce')
order_details['ORDER_TIME'] = pd.to_datetime(order_details['ORDER_TIME'], format='%H:%M:%S', errors='coerce').dt.time

print("Uniques STORE_ID in order details table:" ,order_details.STORE_ID.unique().shape[0])
print("Uniques STORE_ID in store master table:", store_master.STORE_ID.unique().shape[0])

order_details = order_details.merge(
    store_master[['STORE_ID', 'STORE_NAME','PREFIX']],
    on='STORE_ID',
    how='left'
)
print("Unique STORE_NAME in order details after merge:", order_details.STORE_NAME.unique().shape[0])

print("Number of rows with null STORE_NAME:", order_details[order_details['STORE_NAME'].isna()].shape[0])

order_details = order_details.dropna(subset=['STORE_NAME'])
print("Unique STORE_NAME in order details after dropping nulls:", order_details.STORE_NAME.unique().shape[0])

order_details = order_details.merge(
    product_master[[
    'SKU_ID',
    'SKU_NAME',
    'SKU_BRAND',
    'SKU_CATEGORY',
    'SKU_SUBCATEGORY',
    'SKU_RETURNABLE_CONTAINER',
    'SKU_ALCOHOLIC',
    'SKU_OWN_CATALOG',
    'SKU_CONTAINER',
    'SKU_IS_RETURNABLE', 
    'SKU_IS_BUNDLE'
]],
    on='SKU_ID',
    how='left'
)

order_details['SKU_CATEGORY'] = order_details['SKU_CATEGORY'].fillna('Unknown')
order_details['SKU_IS_BUNDLE'] = order_details['SKU_IS_BUNDLE'].fillna(0)

order_details = order_details.merge(
    user_master[['CUSTOMER_ID', 'REGISTRATION_TIMESTAMP', 'CREATION_DATE']],
    on='CUSTOMER_ID',
    how='left'
)

print("Payment method - Value Counts:",order_details.PAYMENT_METHOD.value_counts())

payment_mapping = {
    "CREDIT_CARD_ONLINE": "ONLINE_PAYMENTS",
    "SAFE_PAYMENT_ONLINE": "ONLINE_PAYMENTS",
    "ONLINE_PAYMENTS": "ONLINE_PAYMENTS",
    "CARD ON DELIVERY": "CARD",
    "CARD": "CARD",
    "CASH": "CASH",
    "E WALLET CASHASPP": "E WALLET",
    "E WALLET REVOLUT": "E WALLET",
    "QR CODE": "E WALLET"
}

order_details['PAYMENT_METHOD'] = order_details['PAYMENT_METHOD'].map(payment_mapping)

print("Payment method - Value Counts after mapping:", order_details.PAYMENT_METHOD.value_counts())
print("Order State Final - Value Counts:", order_details.ORDER_STATE_FINAL.value_counts())

order_details = order_details[
    ~order_details['ORDER_STATE_FINAL'].isin(['Fraud'])
]

print("Check Total SKU_NAME null values:", order_details.SKU_NAME.isna().sum())

null_sku_ids = order_details[order_details['SKU_NAME'].isna()]['SKU_ID'].unique()

order_details = order_details.dropna(subset=['SKU_NAME'])
print("Check Total SKU_NAME null values after dropping nulls:", order_details.SKU_NAME.isna().sum())

operation_details['ORDER_DATE'] = pd.to_datetime(operation_details['ORDER_DATE'])
operation_details['ORDER_TIME'] = pd.to_datetime(operation_details['ORDER_TIME'], format='%H:%M:%S', errors='coerce')

operation_details = operation_details.merge(
    order_details[['ORDER_ID', 'CUSTOMER_ID']].drop_duplicates(),
    on='ORDER_ID',
    how='left'
)

# Important query to calculate the discount breakdown columns
discount_breakdown_query = duckdb.query("""
SELECT 
    ORDER_ID
    , ORDER_TIMESTAMP
    , UNIT_QUANTITY
    , UNIT_LIST_PRICE
    , UNIT_OFFERED_PRICE
    , UNIT_CHARGED_PRICE
    , UNIT_DELIVERY_PRICE
    , UNIT_DISCOUNT
    , UNIT_GROSS
    , (UNIT_LIST_PRICE - UNIT_OFFERED_PRICE) AS UNIT_ORGANIC_DISCOUNT
    , (UNIT_OFFERED_PRICE - UNIT_CHARGED_PRICE) AS UNIT_EXTRA_DISCOUNT
    , (UNIT_LIST_PRICE - UNIT_OFFERED_PRICE) * UNIT_QUANTITY AS UNIT_ORGANIC_DISCOUNT_TOTAL
    , (UNIT_OFFERED_PRICE - UNIT_CHARGED_PRICE) * UNIT_QUANTITY AS UNIT_EXTRA_DISCOUNT_TOTAL
    , UNIT_NET
FROM order_details
WHERE true
AND CUSTOMER_ID = '5e9fbcf540e7164b5f8a649a719defcfc53c31ae992418c33793bc2058bd5583'
ORDER BY ORDER_TIMESTAMP DESC
""").to_df()

print("Discount Breakdown Query:")
print(discount_breakdown_query.head())

# Compute discount columns
order_details['UNIT_ORGANIC_DISCOUNT'] = order_details['UNIT_LIST_PRICE'] - order_details['UNIT_OFFERED_PRICE']
order_details['UNIT_EXTRA_DISCOUNT'] = order_details['UNIT_OFFERED_PRICE'] - order_details['UNIT_CHARGED_PRICE']
order_details['UNIT_TOTAL_ORGANIC_DISCOUNT'] = order_details['UNIT_ORGANIC_DISCOUNT'] * order_details['UNIT_QUANTITY']
order_details['UNIT_TOTAL_EXTRA_DISCOUNT'] = order_details['UNIT_EXTRA_DISCOUNT'] * order_details['UNIT_QUANTITY']
order_details['UNIT_TOTAL_DISCOUNTS'] = order_details['UNIT_TOTAL_ORGANIC_DISCOUNT'] + order_details['UNIT_TOTAL_EXTRA_DISCOUNT']


print("Operation Details Final Shape:", operation_details.shape)
print("Final Order Details Shape:", order_details.shape)
print("Valid Orders:", order_details[order_details['ORDER_STATE_FINAL']=='Valid'].shape[0])
print("Invalid Orders:", order_details[order_details['ORDER_STATE_FINAL']=='Invalid'].shape[0])

valid_orders = order_details[order_details['ORDER_STATE_FINAL']=='Valid']

print("Unique custmers with valid orders:", valid_orders.CUSTOMER_ID.unique().shape[0])

# ===========================================
# 3. EXTENDED RFM METRICS CALCULATION
# ===========================================

print("Calculating RFM extended metrics...")
reference_date = order_details['ORDER_DATE'].max()

# RFM & Discount Metrics
rfm_df = valid_orders.groupby(['CUSTOMER_ID']).agg(
    Recency=('ORDER_DATE', lambda x: (reference_date - x.max()).days),
    Frequency=('ORDER_ID', 'nunique'),
    Monetary=('UNIT_NET', 'sum'),
    Total_Units=('UNIT_QUANTITY', 'sum'),
    Total_Organic_Discount=('UNIT_TOTAL_ORGANIC_DISCOUNT', 'sum'),
    Total_Extra_Discount=('UNIT_TOTAL_EXTRA_DISCOUNT', 'sum'),
    Total_Discounts=('UNIT_TOTAL_DISCOUNTS', 'sum')
).reset_index()


# Step 1: Aggregate per ORDER_ID and CUSTOMER_ID to get per-order metrics first
order_level = valid_orders.groupby(['CUSTOMER_ID', 'ORDER_ID']).agg(
    # Discounts aggregation
    ORDER_TOTAL_ORGANIC_DISCOUNT=('UNIT_TOTAL_ORGANIC_DISCOUNT', 'sum'),
    ORDER_TOTAL_ORGANIC_USED_DISCOUNT=('UNIT_TOTAL_ORGANIC_DISCOUNT', lambda x: int(x.sum() > 0)),
    ORDER_TOTAL_EXTRA_DISCOUNT=('UNIT_TOTAL_EXTRA_DISCOUNT','sum'),
    ORDER_TOTAL_EXTRA_USED_DISCOUNT=('UNIT_TOTAL_EXTRA_DISCOUNT', lambda x: int(x.sum() > 0)),
    ORDER_TOTAL_DISCOUNT=('UNIT_TOTAL_DISCOUNTS', 'sum'),
    ORDER_TOTAL_USED_DISCOUNT=('UNIT_TOTAL_DISCOUNTS', lambda x: int(x.sum() > 0)),
    # SKU metrics
    ORDER_UNIT_QUANTITY=('UNIT_QUANTITY', 'sum'),
    ORDER_IS_BULK=('UNIT_QUANTITY', lambda x: int(x.sum() > 2)),
    ORDER_IS_BUNDLE=('SKU_IS_BUNDLE', lambda x: (x == 1).any()),
    # OTHER METRICS
    STORE=('STORE_NAME', lambda x: x.unique()[0]),
    STORE_TYPE=('PREFIX', lambda x: x.unique()[0]),
    PAYMENT_METHOD=('PAYMENT_METHOD', lambda x: x.unique()[0])
).reset_index()

# Step 2: Aggregate per CUSTOMER_ID from per-order metrics
behavioral_df = order_level.groupby('CUSTOMER_ID').agg(
    # Discounts aggregatins per customer
    AVG_ORGANIC_DISCOUNT_PER_ORDER=('ORDER_TOTAL_ORGANIC_DISCOUNT', 'mean'),
    USED_ORGANIC_DISCOUNT_RATIO=('ORDER_TOTAL_ORGANIC_USED_DISCOUNT', 'mean'),
    AVG_EXTRA_DISCOUNT_PER_ORDER=('ORDER_TOTAL_EXTRA_DISCOUNT', 'mean'),
    USED_EXTRA_DISCOUNT_RATIO=('ORDER_TOTAL_EXTRA_USED_DISCOUNT', 'mean'),
    AVG_TOTAL_DISCOUNT_PER_ORDER=('ORDER_TOTAL_DISCOUNT', 'mean'),
    USED_TOTAL_DISCOUNT_RATIO=('ORDER_TOTAL_USED_DISCOUNT', 'mean'),
    # SKU metrics per customer
    AVG_UNIT_QUANTITY_PER_ORDER=('ORDER_UNIT_QUANTITY', 'mean'),
    IS_BULK_BUYER=('ORDER_IS_BULK', 'mean'),
    BUNDLE_PURCHASE_RATIO=('ORDER_IS_BUNDLE', 'mean'),
    # OTHER METRICS
    FAVOURITE_STORE=('STORE', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
    FAVOURITE_STORE_TYPE=('STORE_TYPE', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
    FAVOURITE_PAYMENT_METHOD=('PAYMENT_METHOD', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
).reset_index()

# Diversity and favorite "Categorical Features"
category_features = valid_orders.groupby('CUSTOMER_ID').agg(
    CATEGORY_DIVERSITY=('SKU_CATEGORY', pd.Series.nunique),
    FAVORITE_CATEGORY=('SKU_CATEGORY', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
    SUB_CATEGORY_DIVERSITY=('SKU_SUBCATEGORY', pd.Series.nunique),
    FAVORITE_SUB_CATEGORY=('SKU_SUBCATEGORY', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
).reset_index()

behavioral_df = behavioral_df.merge(category_features, on='CUSTOMER_ID', how='left')

rfm_df = pd.merge(rfm_df, behavioral_df, on='CUSTOMER_ID', how='left')

# Operational Metrics

operation_details['WEEKDAY'] = operation_details['ORDER_DATE'].dt.day_name()
operation_details['HOUR'] = operation_details['ORDER_TIME'].dt.hour

valid_orders_operations = operation_details[operation_details['ORDER_STATE_FINAL']=='Valid']

operational_features = valid_orders_operations.groupby('CUSTOMER_ID').agg(
    AVG_DELIVERY_TIME=('DELIVERY_TIME', 'mean'),
    SLA_VIOLATION_RATE=('WITHIN_SLA', lambda x: (x == 0).sum() / len(x)),
    AVG_APPROVAL_TIME=('APPROVAL_TIME', 'mean'),
    AVG_TOTAL_DISTANCE=('TOTAL_DISTANCE', 'mean'),
    FAVORITE_WEEKDAY=('WEEKDAY', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'),
    FAVORITE_HOUR=('HOUR', lambda x: x.mode()[0] if not x.mode().empty else -1)
).reset_index()

final_rfm = rfm_df.merge(operational_features, on='CUSTOMER_ID', how='left')

# ===================================================
# 4. EXPORTING EXTENDED RFM DATAFRAME
# ===================================================

print("Exporting RFM DataFrame to Parquet format...")
final_rfm.to_parquet("csv_export/RFM2.parquet", index=False)

print(final_rfm.head())

