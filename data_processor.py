import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import io

def load_and_process_retail_data(file_path="Online Retail.xlsx", lookback_days=14, min_touchpoints=2):
    """
    Loads the Online Retail dataset, cleans it, and infers customer journey touchpoints.
    
    Args:
        file_path (str): Path to the retail dataset file
        lookbook_days (int): Number of days to look back for touchpoints before conversion
        min_touchpoints (int): Minimum number of touchpoints required per customer journey
    
    Returns:
        pd.DataFrame: Processed journey data with customer_id, timestamp, channel, cost, conversion
    """
    try:
        # Try to load the default dataset
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # If no file found, generate sample data for demonstration
            from sample_data_generator import generate_sample_ecommerce_data
            st.info("ℹ️ Using sample dataset for demonstration. Upload your own data for real analysis.")
            df = generate_sample_ecommerce_data(num_customers=300, date_range_days=365)
            
    except FileNotFoundError:
        # Generate sample data if default file not found
        from sample_data_generator import generate_sample_ecommerce_data
        st.info("ℹ️ Using sample dataset for demonstration. Upload your own data for real analysis.")
        df = generate_sample_ecommerce_data(num_customers=300, date_range_days=365)
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return pd.DataFrame()
    
    return process_retail_dataframe(df, lookback_days, min_touchpoints)

def process_uploaded_file(uploaded_file, lookback_days=14, min_touchpoints=2):
    """
    Process an uploaded file and convert it to journey data.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        lookback_days (int): Number of days to look back for touchpoints
        min_touchpoints (int): Minimum number of touchpoints required
    
    Returns:
        pd.DataFrame: Processed journey data
    """
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
            return pd.DataFrame()
        
        return process_retail_dataframe(df, lookback_days, min_touchpoints)
        
    except Exception as e:
        st.error(f"❌ Error processing uploaded file: {str(e)}")
        return pd.DataFrame()

def process_retail_dataframe(df, lookback_days=14, min_touchpoints=2):
    """
    Process a retail dataframe and convert it to customer journey data.
    
    Args:
        df (pd.DataFrame): Raw retail transaction data
        lookback_days (int): Number of days to look back for touchpoints
        min_touchpoints (int): Minimum number of touchpoints required
    
    Returns:
        pd.DataFrame: Processed journey data
    """
    try:
        # Validate required columns
        required_columns = ['CustomerID', 'InvoiceDate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Try alternative column names
            column_mapping = {
                'customer_id': 'CustomerID',
                'Customer_ID': 'CustomerID',
                'customerId': 'CustomerID',
                'invoice_date': 'InvoiceDate',
                'Invoice_Date': 'InvoiceDate',
                'invoiceDate': 'InvoiceDate',
                'date': 'InvoiceDate',
                'Date': 'InvoiceDate',
                'timestamp': 'InvoiceDate',
                'Timestamp': 'InvoiceDate'
            }
            
            # Try to map alternative column names
            for alt_name, standard_name in column_mapping.items():
                if alt_name in df.columns and standard_name not in df.columns:
                    df = df.rename(columns={alt_name: standard_name})
            
            # Check again for required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}. Please ensure your dataset has CustomerID and InvoiceDate columns.")
                return pd.DataFrame()
        
        # Clean the data
        df_clean = clean_retail_data(df)
        
        if df_clean.empty:
            st.error("❌ No valid data remaining after cleaning.")
            return pd.DataFrame()
        
        # Generate customer journeys with inferred touchpoints
        journey_data = generate_customer_journeys(df_clean, lookback_days, min_touchpoints)
        
        return journey_data
        
    except Exception as e:
        st.error(f"❌ Error processing retail dataframe: {str(e)}")
        return pd.DataFrame()

def clean_retail_data(df):
    """
    Clean the retail transaction data.
    
    Args:
        df (pd.DataFrame): Raw retail data
    
    Returns:
        pd.DataFrame: Cleaned retail data
    """
    try:
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Remove rows with missing CustomerID
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=['CustomerID'])
        
        if len(df_clean) < initial_rows:
            st.info(f"ℹ️ Removed {initial_rows - len(df_clean)} rows with missing CustomerID")
        
        # Convert InvoiceDate to datetime
        if df_clean['InvoiceDate'].dtype == 'object':
            df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
        
        # Remove rows with invalid dates
        df_clean = df_clean.dropna(subset=['InvoiceDate'])
        
        # Remove duplicate transactions (same customer, same date, same invoice if available)
        duplicate_cols = ['CustomerID', 'InvoiceDate']
        if 'InvoiceNo' in df_clean.columns:
            duplicate_cols.append('InvoiceNo')
        
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=duplicate_cols)
        
        if len(df_clean) < before_dedup:
            st.info(f"ℹ️ Removed {before_dedup - len(df_clean)} duplicate transactions")
        
        # Handle negative quantities/prices if present (returns/corrections)
        if 'Quantity' in df_clean.columns:
            df_clean = df_clean[df_clean['Quantity'] > 0]
        
        if 'UnitPrice' in df_clean.columns:
            df_clean = df_clean[df_clean['UnitPrice'] > 0]
        
        # Sort by customer and date
        df_clean = df_clean.sort_values(['CustomerID', 'InvoiceDate'])
        
        return df_clean
        
    except Exception as e:
        st.error(f"❌ Error cleaning data: {str(e)}")
        return pd.DataFrame()

def generate_customer_journeys(df, lookback_days=14, min_touchpoints=2):
    """
    Generate customer journey data with inferred marketing touchpoints.
    
    Args:
        df (pd.DataFrame): Cleaned retail transaction data
        lookback_days (int): Number of days to look back for touchpoints
        min_touchpoints (int): Minimum number of touchpoints required
    
    Returns:
        pd.DataFrame: Customer journey data with touchpoints
    """
    try:
        # Define marketing channels for touchpoint inference
        marketing_channels = [
            'Direct',
            'Organic Search', 
            'Paid Search',
            'Display Ad',
            'Social Media',
            'Email Campaign'
        ]
        
        # Create journey data list
        journey_records = []
        
        # Process each customer
        customers = df['CustomerID'].unique()
        
        for customer_id in customers:
            customer_data = df[df['CustomerID'] == customer_id].sort_values('InvoiceDate')
            
            # Each unique invoice date represents a conversion event
            conversion_dates = customer_data['InvoiceDate'].unique()
            
            for conversion_date in conversion_dates:
                # Add the conversion touchpoint (actual purchase)
                journey_records.append({
                    'customer_id': customer_id,
                    'timestamp': conversion_date,
                    'channel': 'Direct',  # Assume direct for actual purchase
                    'cost': 0.0,  # No cost for conversion event
                    'conversion': True
                })
                
                # Generate pre-conversion touchpoints
                touchpoints_generated = generate_pre_conversion_touchpoints(
                    customer_id, 
                    conversion_date, 
                    lookback_days, 
                    marketing_channels
                )
                
                journey_records.extend(touchpoints_generated)
        
        # Convert to DataFrame
        journey_df = pd.DataFrame(journey_records)
        
        if journey_df.empty:
            return journey_df
        
        # Sort by customer and timestamp
        journey_df = journey_df.sort_values(['customer_id', 'timestamp'])
        
        # Filter customers with minimum touchpoints
        customer_touchpoint_counts = journey_df.groupby('customer_id').size()
        valid_customers = customer_touchpoint_counts[customer_touchpoint_counts >= min_touchpoints].index
        journey_df = journey_df[journey_df['customer_id'].isin(valid_customers)]
        
        # Reset index
        journey_df = journey_df.reset_index(drop=True)
        
        return journey_df
        
    except Exception as e:
        st.error(f"❌ Error generating customer journeys: {str(e)}")
        return pd.DataFrame()

def generate_pre_conversion_touchpoints(customer_id, conversion_date, lookback_days, channels):
    """
    Generate realistic pre-conversion touchpoints for a customer.
    
    Args:
        customer_id: Unique customer identifier
        conversion_date: Date of conversion
        lookback_days (int): Number of days to look back
        channels (list): Available marketing channels
    
    Returns:
        list: List of touchpoint records
    """
    touchpoints = []
    
    # Generate 1-3 touchpoints before conversion
    num_touchpoints = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
    
    # Define typical customer journey patterns
    journey_patterns = [
        ['Display Ad', 'Paid Search'],
        ['Social Media', 'Organic Search'],
        ['Email Campaign', 'Direct'],
        ['Display Ad', 'Social Media', 'Paid Search'],
        ['Organic Search'],
        ['Paid Search']
    ]
    
    # Select a journey pattern
    if num_touchpoints == 1:
        selected_channels = [np.random.choice(['Organic Search', 'Paid Search', 'Social Media'])]
    elif num_touchpoints == 2:
        pattern = np.random.choice(journey_patterns[:3])
        selected_channels = pattern[:2]
    else:
        pattern = journey_patterns[3]  # 3-touchpoint pattern
        selected_channels = pattern
    
    # Generate touchpoints with realistic timing
    for i, channel in enumerate(selected_channels):
        # Generate timestamp (days before conversion)
        if i == 0:  # First touchpoint (furthest from conversion)
            days_before = np.random.randint(max(1, lookback_days-7), lookback_days)
        elif i == 1:  # Second touchpoint
            days_before = np.random.randint(1, max(2, lookback_days//2))
        else:  # Third touchpoint (closest to conversion)
            days_before = np.random.randint(1, 3)
        
        touchpoint_date = conversion_date - timedelta(days=days_before)
        
        # Generate realistic cost based on channel
        cost = generate_channel_cost(channel)
        
        touchpoints.append({
            'customer_id': customer_id,
            'timestamp': touchpoint_date,
            'channel': channel,
            'cost': cost,
            'conversion': False
        })
    
    return touchpoints

def generate_channel_cost(channel):
    """
    Generate realistic cost for a marketing channel touchpoint.
    
    Args:
        channel (str): Marketing channel name
    
    Returns:
        float: Cost for the touchpoint
    """
    # Define cost ranges for different channels (in dollars)
    cost_ranges = {
        'Display Ad': (0.50, 3.00),
        'Paid Search': (1.00, 5.00),
        'Social Media': (0.25, 2.00),
        'Email Campaign': (0.05, 0.30),
        'Organic Search': (0.00, 0.00),  # No direct cost
        'Direct': (0.00, 0.00)  # No direct cost
    }
    
    min_cost, max_cost = cost_ranges.get(channel, (0.50, 2.00))
    
    if min_cost == max_cost == 0.00:
        return 0.00
    
    return round(np.random.uniform(min_cost, max_cost), 2)

def validate_journey_data(journey_df):
    """
    Validate the structure and quality of journey data.
    
    Args:
        journey_df (pd.DataFrame): Journey data to validate
    
    Returns:
        bool: True if data is valid, False otherwise
    """
    required_columns = ['customer_id', 'timestamp', 'channel', 'cost', 'conversion']
    
    # Check required columns
    if not all(col in journey_df.columns for col in required_columns):
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(journey_df['timestamp']):
        return False
    
    # Check for valid conversions
    if not journey_df['conversion'].dtype == bool:
        return False
    
    # Check that each customer has at least one conversion
    customers_with_conversions = journey_df[journey_df['conversion'] == True]['customer_id'].unique()
    total_customers = journey_df['customer_id'].nunique()
    
    if len(customers_with_conversions) == 0:
        return False
    
    return True
