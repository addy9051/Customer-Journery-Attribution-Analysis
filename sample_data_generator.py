import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_ecommerce_data(num_customers=500, date_range_days=365):
    """
    Generate realistic e-commerce transaction data for testing the attribution platform.
    
    Args:
        num_customers (int): Number of unique customers to generate
        date_range_days (int): Number of days to span the data over
    
    Returns:
        pd.DataFrame: Synthetic e-commerce transaction data
    """
    
    # Set random seed for reproducible results
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic product categories and price ranges
    products = {
        'Electronics': {'price_range': (50, 800), 'frequency': 0.3},
        'Clothing': {'price_range': (20, 200), 'frequency': 0.25},
        'Home & Garden': {'price_range': (15, 300), 'frequency': 0.2},
        'Books': {'price_range': (10, 50), 'frequency': 0.15},
        'Sports & Outdoors': {'price_range': (25, 400), 'frequency': 0.1}
    }
    
    # Define countries with realistic distribution
    countries = {
        'United Kingdom': 0.7,
        'Germany': 0.1,
        'France': 0.08,
        'Netherlands': 0.05,
        'Spain': 0.04,
        'Italy': 0.03
    }
    
    # Generate customer base
    customer_ids = range(10000, 10000 + num_customers)
    
    # Define customer behavior patterns
    customer_behaviors = {
        'frequent_buyer': {'transaction_freq': (20, 40), 'avg_items': (2, 5), 'probability': 0.1},
        'regular_buyer': {'transaction_freq': (5, 15), 'avg_items': (1, 3), 'probability': 0.3},
        'occasional_buyer': {'transaction_freq': (2, 8), 'avg_items': (1, 2), 'probability': 0.4},
        'one_time_buyer': {'transaction_freq': (1, 2), 'avg_items': (1, 1), 'probability': 0.2}
    }
    
    transactions = []
    
    # Start and end dates
    start_date = datetime.now() - timedelta(days=date_range_days)
    end_date = datetime.now()
    
    for customer_id in customer_ids:
        # Assign customer behavior
        behavior_type = np.random.choice(
            list(customer_behaviors.keys()),
            p=[customer_behaviors[bt]['probability'] for bt in customer_behaviors.keys()]
        )
        behavior = customer_behaviors[behavior_type]
        
        # Assign customer country
        customer_country = np.random.choice(
            list(countries.keys()),
            p=list(countries.values())
        )
        
        # Determine number of transactions for this customer
        num_transactions = np.random.randint(
            behavior['transaction_freq'][0],
            behavior['transaction_freq'][1] + 1
        )
        
        # Generate transaction dates
        transaction_dates = []
        for _ in range(num_transactions):
            # Generate random date within range
            random_days = np.random.randint(0, date_range_days)
            transaction_date = start_date + timedelta(days=random_days)
            transaction_dates.append(transaction_date)
        
        # Sort dates chronologically
        transaction_dates.sort()
        
        # Generate transactions for this customer
        invoice_counter = 500000
        for transaction_date in transaction_dates:
            # Number of items in this transaction
            num_items = np.random.randint(
                behavior['avg_items'][0],
                behavior['avg_items'][1] + 1
            )
            
            invoice_no = f"INV{invoice_counter}"
            invoice_counter += 1
            
            for item_num in range(num_items):
                # Select product category
                category = np.random.choice(
                    list(products.keys()),
                    p=[products[cat]['frequency'] for cat in products.keys()]
                )
                
                # Generate product details
                price_range = products[category]['price_range']
                unit_price = round(np.random.uniform(price_range[0], price_range[1]), 2)
                
                # Generate quantity (most transactions are single items)
                quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.06, 0.04])
                
                # Generate stock code (realistic format)
                stock_code = f"{category[:3].upper()}{np.random.randint(1000, 9999)}"
                
                # Generate description
                descriptions = {
                    'Electronics': ['Wireless Headphones', 'Smartphone Case', 'USB Cable', 'Power Bank', 'Tablet Stand'],
                    'Clothing': ['Cotton T-Shirt', 'Denim Jeans', 'Winter Jacket', 'Running Shoes', 'Casual Dress'],
                    'Home & Garden': ['Coffee Mug', 'Kitchen Knife', 'Garden Tool', 'Decorative Vase', 'Storage Box'],
                    'Books': ['Fiction Novel', 'Cookbook', 'Business Guide', 'Travel Guide', 'Art Book'],
                    'Sports & Outdoors': ['Yoga Mat', 'Water Bottle', 'Hiking Backpack', 'Tennis Racket', 'Camping Gear']
                }
                
                description = np.random.choice(descriptions[category])
                
                # Add some time variation within the day
                hours = np.random.randint(8, 22)
                minutes = np.random.randint(0, 60)
                transaction_datetime = transaction_date.replace(hour=hours, minute=minutes)
                
                transactions.append({
                    'InvoiceNo': invoice_no,
                    'StockCode': stock_code,
                    'Description': description,
                    'Quantity': quantity,
                    'InvoiceDate': transaction_datetime,
                    'UnitPrice': unit_price,
                    'CustomerID': customer_id,
                    'Country': customer_country
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Add some realistic data patterns
    
    # 1. Seasonal patterns (higher sales in November-December)
    df['Month'] = df['InvoiceDate'].dt.month
    holiday_months = [11, 12]
    
    # Increase transaction volume for holiday months
    holiday_mask = df['Month'].isin(holiday_months)
    holiday_customers = df[holiday_mask]['CustomerID'].unique()
    
    # Add extra transactions for holiday period
    extra_transactions = []
    if len(holiday_customers) > 0:
        selected_customers = np.random.choice(holiday_customers, size=max(1, len(holiday_customers)//2), replace=False)
        for customer_id in selected_customers:
            # Add 1-2 extra holiday transactions
            for _ in range(np.random.randint(1, 3)):
                holiday_date = datetime(2024, 12, np.random.randint(1, 25))
                
                # Generate holiday transaction
                category = np.random.choice(list(products.keys()))
                price_range = products[category]['price_range']
                unit_price = round(np.random.uniform(price_range[0], price_range[1]), 2)
                
                customer_country = df[df['CustomerID'] == customer_id]['Country'].iloc[0]
                
                extra_transactions.append({
                    'InvoiceNo': f"INV{invoice_counter}",
                    'StockCode': f"{category[:3].upper()}{np.random.randint(1000, 9999)}",
                    'Description': f"Holiday {category} Item",
                    'Quantity': np.random.randint(1, 3),
                    'InvoiceDate': holiday_date,
                    'UnitPrice': unit_price,
                    'CustomerID': customer_id,
                    'Country': customer_country
                })
                invoice_counter += 1
    
    if extra_transactions:
        extra_df = pd.DataFrame(extra_transactions)
        df = pd.concat([df, extra_df], ignore_index=True)
    
    # 2. Add some returns/cancellations (negative quantities)
    num_returns = len(df) // 50  # About 2% return rate
    return_indices = np.random.choice(df.index, size=num_returns, replace=False)
    
    for idx in return_indices:
        # Create return transaction
        original = df.loc[idx].copy()
        return_transaction = {
            'InvoiceNo': f"C{original['InvoiceNo'][3:]}",  # Return invoice format
            'StockCode': original['StockCode'],
            'Description': original['Description'],
            'Quantity': -abs(original['Quantity']),  # Negative quantity
            'InvoiceDate': original['InvoiceDate'] + timedelta(days=np.random.randint(1, 30)),
            'UnitPrice': original['UnitPrice'],
            'CustomerID': original['CustomerID'],
            'Country': original['Country']
        }
        df = pd.concat([df, pd.DataFrame([return_transaction])], ignore_index=True)
    
    # Sort by date and customer
    df = df.sort_values(['CustomerID', 'InvoiceDate'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Remove the temporary Month column
    df = df.drop('Month', axis=1)
    
    return df

def save_sample_dataset(filename="sample_ecommerce_data.csv", num_customers=500):
    """
    Generate and save a sample e-commerce dataset.
    
    Args:
        filename (str): Output filename
        num_customers (int): Number of customers to generate
    """
    print(f"Generating sample e-commerce dataset with {num_customers} customers...")
    
    # Generate the data
    df = generate_sample_ecommerce_data(num_customers=num_customers)
    
    # Save to file
    if filename.endswith('.csv'):
        df.to_csv(filename, index=False)
    elif filename.endswith('.xlsx'):
        df.to_excel(filename, index=False)
    else:
        # Default to CSV
        filename += '.csv'
        df.to_csv(filename, index=False)
    
    print(f"Sample dataset saved as '{filename}'")
    print(f"Dataset contains {len(df)} transactions for {df['CustomerID'].nunique()} customers")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    
    return df

if __name__ == "__main__":
    # Generate and save sample data when script is run directly
    df = save_sample_dataset("sample_ecommerce_data.csv", num_customers=300)
    print("\nDataset preview:")
    print(df.head(10))
    print("\nDataset info:")
    print(df.info())