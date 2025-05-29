import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def first_touch_attribution(journey_data):
    """
    Implements first-touch attribution model.
    Credits the first touchpoint in each customer's journey with the full conversion.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data with columns:
                                   customer_id, timestamp, channel, cost, conversion
    
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    try:
        # Group by customer and find first touchpoint for each customer journey
        first_touch_data = journey_data.sort_values(['customer_id', 'timestamp']).groupby('customer_id').first()
        
        # Count conversions by first-touch channel
        # Only count customers who actually converted
        converting_customers = journey_data[journey_data['conversion']==True]['customer_id'].unique()
        first_touch_converting = first_touch_data[first_touch_data.index.isin(converting_customers)]
        
        attribution_results = first_touch_converting['channel'].value_counts().reset_index()
        attribution_results.columns = ['channel', 'attributed_conversions']
        
        return attribution_results
        
    except Exception as e:
        print(f"Error in first_touch_attribution: {str(e)}")
        return pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))

def last_touch_attribution(journey_data):
    """
    Implements last-touch attribution model.
    Credits the last touchpoint before conversion with the full conversion.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
    
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    try:
        attribution_results = []
        
        # Get all customers who converted
        converting_customers = journey_data[journey_data['conversion']==True]['customer_id'].unique()
        
        for customer_id in converting_customers:
            customer_journey = journey_data[journey_data['customer_id'] == customer_id].sort_values('timestamp')
            
            # Find conversion events for this customer
            conversion_events = customer_journey[customer_journey['conversion']]
            
            for _, conversion_event in conversion_events.iterrows():
                conversion_timestamp = conversion_event['timestamp']
                
                # Find last touchpoint before this conversion
                pre_conversion_touchpoints = customer_journey[
                    (customer_journey['timestamp'] <= conversion_timestamp) & 
                    (customer_journey['conversion'] == False)
                ]
                
                if not pre_conversion_touchpoints.empty:
                    # Get the last non-conversion touchpoint
                    last_touchpoint = pre_conversion_touchpoints.iloc[-1]
                    attribution_results.append({
                        'channel': last_touchpoint['channel'],
                        'attributed_conversions': 1.0
                    })
                else:
                    # If no pre-conversion touchpoints, attribute to the conversion touchpoint itself
                    attribution_results.append({
                        'channel': conversion_event['channel'],
                        'attributed_conversions': 1.0
                    })
        
        # Aggregate results
        if attribution_results:
            results_df = pd.DataFrame(attribution_results)
            final_results = results_df.groupby('channel')['attributed_conversions'].sum().reset_index()
        else:
            final_results = pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))
        
        return final_results
        
    except Exception as e:
        print(f"Error in last_touch_attribution: {str(e)}")
        return pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))

def linear_attribution(journey_data):
    """
    Implements linear attribution model.
    Distributes conversion credit equally across all touchpoints in the customer journey.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
    
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    try:
        attribution_results = []
        
        # Get all customers who converted
        converting_customers = journey_data[journey_data['conversion']==True]['customer_id'].unique()
        
        for customer_id in converting_customers:
            customer_journey = journey_data[journey_data['customer_id'] == customer_id].sort_values('timestamp')
            
            # Find conversion events for this customer
            conversion_events = customer_journey[customer_journey['conversion']==True]
            
            for _, conversion_event in conversion_events.iterrows():
                conversion_timestamp = conversion_event['timestamp']
                
                # Find all touchpoints for this conversion journey
                # Include touchpoints up to and including the conversion
                journey_touchpoints = customer_journey[customer_journey['timestamp'] <= conversion_timestamp]
                
                # Calculate equal attribution for each touchpoint
                num_touchpoints = len(journey_touchpoints)
                if num_touchpoints > 0:
                    attribution_per_touchpoint = 1.0 / num_touchpoints
                    
                    for _, touchpoint in journey_touchpoints.iterrows():
                        attribution_results.append({
                            'channel': touchpoint['channel'],
                            'attributed_conversions': attribution_per_touchpoint
                        })
        
        # Aggregate results
        if attribution_results:
            results_df = pd.DataFrame(attribution_results)
            final_results = results_df.groupby('channel')['attributed_conversions'].sum().reset_index()
        else:
            final_results = pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))
        
        return final_results
        
    except Exception as e:
        print(f"Error in linear_attribution: {str(e)}")
        return pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))

def time_decay_attribution(journey_data, decay_rate=0.5):
    """
    Implements time-decay attribution model.
    Gives more credit to touchpoints closer to conversion.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
        decay_rate (float): Rate at which attribution decays with time (0-1)
    
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    try:
        attribution_results = []
        
        # Get all customers who converted
        converting_customers = journey_data[journey_data['conversion'] == True]['customer_id'].unique()
        
        for customer_id in converting_customers:
            customer_journey = journey_data[journey_data['customer_id'] == customer_id].sort_values('timestamp')
            
            # Find conversion events for this customer
            conversion_events = customer_journey[customer_journey['conversion'] == True]
            
            for _, conversion_event in conversion_events.iterrows():
                conversion_timestamp = conversion_event['timestamp']
                
                # Find all touchpoints for this conversion journey
                journey_touchpoints = customer_journey[customer_journey['timestamp'] <= conversion_timestamp]
                
                if len(journey_touchpoints) > 0:
                    # Calculate time-based weights
                    weights = []
                    for _, touchpoint in journey_touchpoints.iterrows():
                        # Calculate days between touchpoint and conversion
                        time_diff = (conversion_timestamp - touchpoint['timestamp']).days
                        # Apply exponential decay
                        weight = decay_rate ** time_diff
                        weights.append(weight)
                    
                    # Normalize weights to sum to 1
                    total_weight = sum(weights)
                    if total_weight > 0:
                        normalized_weights = [w / total_weight for w in weights]
                        
                        # Assign attribution
                        for i, (_, touchpoint) in enumerate(journey_touchpoints.iterrows()):
                            attribution_results.append({
                                'channel': touchpoint['channel'],
                                'attributed_conversions': normalized_weights[i]
                            })
        
        # Aggregate results
        if attribution_results:
            results_df = pd.DataFrame(attribution_results)
            final_results = results_df.groupby('channel')['attributed_conversions'].sum().reset_index()
        else:
            final_results = pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))
        
        return final_results
        
    except Exception as e:
        print(f"Error in time_decay_attribution: {str(e)}")
        return pd.DataFrame(columns=pd.Index((['channel', 'attributed_conversions'])))

def position_based_attribution(journey_data, first_touch_weight=0.4, last_touch_weight=0.4):
    """
    Implements position-based (U-shaped) attribution model.
    Gives more credit to first and last touchpoints, remaining credit distributed equally.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
        first_touch_weight (float): Weight for first touchpoint (default 40%)
        last_touch_weight (float): Weight for last touchpoint (default 40%)
    
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    try:
        attribution_results = []
        middle_weight = 1.0 - first_touch_weight - last_touch_weight
        
        # Get all customers who converted
        converting_customers = journey_data[journey_data['conversion'] == True]['customer_id'].unique()
        
        for customer_id in converting_customers:
            customer_journey = journey_data[journey_data['customer_id'] == customer_id].sort_values('timestamp')
            
            # Find conversion events for this customer
            conversion_events = customer_journey[customer_journey['conversion'] == True]
            
            for _, conversion_event in conversion_events.iterrows():
                conversion_timestamp = conversion_event['timestamp']
                
                # Find all touchpoints for this conversion journey
                journey_touchpoints = customer_journey[customer_journey['timestamp'] <= conversion_timestamp]
                
                num_touchpoints = len(journey_touchpoints)
                
                if num_touchpoints == 1:
                    # Single touchpoint gets full credit
                    touchpoint = journey_touchpoints.iloc[0]
                    attribution_results.append({
                        'channel': touchpoint['channel'],
                        'attributed_conversions': 1.0
                    })
                elif num_touchpoints == 2:
                    # Two touchpoints: split between first and last
                    first_touchpoint = journey_touchpoints.iloc[0]
                    last_touchpoint = journey_touchpoints.iloc[-1]
                    
                    attribution_results.append({
                        'channel': first_touchpoint['channel'],
                        'attributed_conversions': first_touch_weight + (middle_weight / 2)
                    })
                    attribution_results.append({
                        'channel': last_touchpoint['channel'],
                        'attributed_conversions': last_touch_weight + (middle_weight / 2)
                    })
                else:
                    # Multiple touchpoints: first, middle, last
                    first_touchpoint = journey_touchpoints.iloc[0]
                    last_touchpoint = journey_touchpoints.iloc[-1]
                    middle_touchpoints = journey_touchpoints.iloc[1:-1]
                    
                    # First touchpoint
                    attribution_results.append({
                        'channel': first_touchpoint['channel'],
                        'attributed_conversions': first_touch_weight
                    })
                    
                    # Last touchpoint
                    attribution_results.append({
                        'channel': last_touchpoint['channel'],
                        'attributed_conversions': last_touch_weight
                    })
                    
                    # Middle touchpoints (equal distribution)
                    if len(middle_touchpoints) > 0:
                        middle_attribution_per_touchpoint = middle_weight / len(middle_touchpoints)
                        for _, touchpoint in middle_touchpoints.iterrows():
                            attribution_results.append({
                                'channel': touchpoint['channel'],
                                'attributed_conversions': middle_attribution_per_touchpoint
                            })
        
        # Aggregate results
        if attribution_results:
            results_df = pd.DataFrame(attribution_results)
            final_results = results_df.groupby('channel')['attributed_conversions'].sum().reset_index()
        else:
            final_results = pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))
        
        return final_results
        
    except Exception as e:
        print(f"Error in position_based_attribution: {str(e)}")
        return pd.DataFrame(columns=pd.Index(['channel', 'attributed_conversions']))

def calculate_attribution_comparison(journey_data):
    """
    Calculate attribution results for all models and return comparison data.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
    
    Returns:
        pd.DataFrame: Comparison results with columns: model, channel, attributed_conversions
    """
    try:
        comparison_results = []
        
        # Calculate attribution for each model
        models = {
            'First-Touch': first_touch_attribution,
            'Last-Touch': last_touch_attribution,
            'Linear': linear_attribution,
            'Time-Decay': time_decay_attribution,
            'Position-Based': position_based_attribution
        }
        
        for model_name, attribution_function in models.items():
            try:
                if model_name in ['Time-Decay', 'Position-Based']:
                    # These models have additional parameters
                    if model_name == 'Time-Decay':
                        results = attribution_function(journey_data, decay_rate=0.5)
                    else:  # Position-Based
                        results = attribution_function(journey_data)
                else:
                    results = attribution_function(journey_data)
                
                # Add model name to results
                results['model'] = model_name
                comparison_results.append(results)
                
            except Exception as e:
                print(f"Error calculating {model_name} attribution: {str(e)}")
                continue
        
        # Combine all results
        if comparison_results:
            final_comparison = pd.concat(comparison_results, ignore_index=True)
            return final_comparison
        else:
            return pd.DataFrame(columns=pd.Index(['model', 'channel', 'attributed_conversions']))
        
    except Exception as e:
        print(f"Error in calculate_attribution_comparison: {str(e)}")
        return pd.DataFrame(columns=pd.Index(['model', 'channel', 'attributed_conversions']))

def calculate_attribution_metrics(journey_data, attribution_results):
    """
    Calculate additional metrics for attribution analysis.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
        attribution_results (pd.DataFrame): Attribution results
    
    Returns:
        dict: Dictionary with various attribution metrics
    """
    try:
        metrics = {}
        
        # Total conversions
        total_conversions = len(journey_data[journey_data['conversion'] == True])
        metrics['total_conversions'] = total_conversions
        
        # Total attributed conversions (should equal total conversions)
        total_attributed = attribution_results['attributed_conversions'].sum()
        metrics['total_attributed_conversions'] = total_attributed
        
        # Channel performance metrics
        channel_metrics = []
        for _, row in attribution_results.iterrows():
            channel = row['channel']
            attributed_conversions = row['attributed_conversions']
            
            # Calculate channel-specific metrics
            channel_data = journey_data[journey_data['channel'] == channel]
            total_touchpoints = len(channel_data)
            total_cost = channel_data['cost'].sum()
            
            channel_metric = {
                'channel': channel,
                'attributed_conversions': attributed_conversions,
                'total_touchpoints': total_touchpoints,
                'total_cost': total_cost,
                'cost_per_conversion': total_cost / attributed_conversions if attributed_conversions > 0 else 0,
                'attribution_percentage': (attributed_conversions / total_attributed * 100) if total_attributed > 0 else 0
            }
            channel_metrics.append(channel_metric)
        
        metrics['channel_metrics'] = pd.DataFrame(channel_metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating attribution metrics: {str(e)}")
        return {}
