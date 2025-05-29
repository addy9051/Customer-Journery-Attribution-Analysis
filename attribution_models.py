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

def calculate_attribution_comparison(journey_data, decay_rate=0.5, first_touch_weight=0.4, last_touch_weight=0.4):
    """
    Calculate attribution results for all models and return comparison data.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data
        decay_rate (float): Decay rate for time-decay attribution
        first_touch_weight (float): Weight for first touch in position-based attribution
        last_touch_weight (float): Weight for last touch in position-based attribution
    
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
            'Position-Based': position_based_attribution,
            'Markov Chain': markov_chain_attribution,
            'Shapley Value': shapley_value_attribution
        }
        
        for model_name, attribution_function in models.items():
            try:
                if model_name == 'Time-Decay':
                    results = attribution_function(journey_data, decay_rate=decay_rate)
                elif model_name == 'Position-Based':
                    results = attribution_function(journey_data, first_touch_weight=first_touch_weight, last_touch_weight=last_touch_weight)
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

def _build_transition_matrix(journey_data, states, state_to_idx, min_journey_length=2, verbose=False):
    """
    Builds a transition matrix from journey data.
    
    Args:
        journey_data: DataFrame containing customer journeys
        states: List of all possible states
        state_to_idx: Dictionary mapping state names to indices
        min_journey_length: Minimum number of touchpoints required
        verbose: Whether to print debug information
        
    Returns:
        Transition probability matrix
    """
    import numpy as np
    
    n_states = len(states)
    transition_counts = np.zeros((n_states, n_states))
    
    # Group by customer and process each journey
    for customer_id, journey in journey_data.groupby('customer_id'):
        # Sort by timestamp
        journey = journey.sort_values('timestamp')
        
        # Get conversion events
        conversion_events = journey[journey['conversion']]
        if conversion_events.empty:
            continue
            
        # Get touchpoints before first conversion
        first_conversion = conversion_events.iloc[0]
        touchpoints = journey[journey['timestamp'] <= first_conversion['timestamp']]
        
        if len(touchpoints) < min_journey_length:
            continue
            
        # Build path with start and end states
        path = ['(start)'] + touchpoints['channel'].tolist() + ['(conversion)']
        
        if verbose and customer_id % 100 == 0:  # Log every 100th customer
            print(f"Customer {customer_id} path: {path}")
        
        # Count transitions
        for i in range(len(path) - 1):
            from_state = path[i]
            to_state = path[i + 1]
            
            if from_state in state_to_idx and to_state in state_to_idx:
                from_idx = state_to_idx[from_state]
                to_idx = state_to_idx[to_state]
                transition_counts[from_idx, to_idx] += 1
    
    # Add small constant to avoid zeros
    transition_matrix = transition_counts + 1e-10
    
    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_probs = np.divide(
        transition_matrix, 
        row_sums,
        out=np.full_like(transition_matrix, 1.0/n_states),
        where=row_sums > 0
    )
    
    return transition_probs

def _calculate_steady_state(transition_probs, max_iterations=1000, tol=1e-6):
    """
    Calculates the steady-state distribution using power iteration.
    
    Args:
        transition_probs: Transition probability matrix
        max_iterations: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        Steady-state probability vector
    """
    import numpy as np
    n = transition_probs.shape[0]
    pi = np.ones(n) / n  # Initial uniform distribution
    
    for _ in range(max_iterations):
        new_pi = pi @ transition_probs
        
        # Check for convergence using L1 norm
        if np.linalg.norm(new_pi - pi, 1) < tol:
            break
            
        pi = new_pi
    
    return pi

def _calculate_removal_effect(transition_probs, states, state_to_idx, max_iterations=1000, tol=1e-6):
    """
    Calculates the removal effect for each channel.
    
    Args:
        transition_probs: Transition probability matrix
        states: List of all states
        state_to_idx: Dictionary mapping state names to indices
        max_iterations: Maximum iterations for convergence
        tol: Convergence tolerance
        
    Returns:
        Dictionary of removal effects by channel
    """
    import numpy as np
    
    # Calculate baseline conversion probability
    steady_state = _calculate_steady_state(transition_probs, max_iterations, tol)
    conv_idx = state_to_idx.get('(conversion)', -1)
    if conv_idx == -1:
        return {state: 0 for state in states if state not in ['(start)', '(conversion)', '(null)']}
    
    baseline_conversion = steady_state[conv_idx]
    
    # Calculate removal effect for each channel
    removal_effects = {}
    channels = [s for s in states if s not in ['(start)', '(conversion)', '(null)']]
    
    for channel in channels:
        if channel not in state_to_idx:
            removal_effects[channel] = 0
            continue
            
        # Create a copy of the transition matrix
        modified_probs = transition_probs.copy()
        channel_idx = state_to_idx[channel]
        
        # Remove the channel by redirecting its transitions to null
        modified_probs[channel_idx, :] = 0
        modified_probs[channel_idx, channel_idx] = 1.0  # Absorbing state
        
        # Re-normalize other transitions
        for i in range(modified_probs.shape[0]):
            row_sum = modified_probs[i, :].sum()
            if row_sum > 0:
                modified_probs[i, :] /= row_sum
        
        # Calculate new conversion probability
        new_steady_state = _calculate_steady_state(modified_probs, max_iterations, tol)
        new_conversion = new_steady_state[conv_idx]
        
        # Calculate removal effect
        if baseline_conversion > 1e-10:  # Avoid division by zero
            removal_effect = (baseline_conversion - new_conversion) / baseline_conversion
            removal_effects[channel] = max(0, removal_effect)  # Ensure non-negative
        else:
            removal_effects[channel] = 0
    
    return removal_effects

def markov_chain_attribution(journey_data, min_journey_length=2, max_iterations=1000, random_state=42, verbose=False):
    """
    Implements Markov Chain attribution model with improved stability and performance.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data with columns:
                                  customer_id, timestamp, channel, conversion
        min_journey_length (int): Minimum number of touchpoints required for analysis
        max_iterations (int): Maximum number of iterations for convergence
        random_state (int): Random seed for reproducibility
        verbose (bool): Whether to print detailed debug information
        
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
    """
    import numpy as np
    import pandas as pd
    
    try:
        if journey_data.empty:
            raise ValueError("Input journey data is empty")
            
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Get unique channels and define states
        channels = journey_data['channel'].unique().tolist()
        states = ['(start)'] + channels + ['(conversion)', '(null)']
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        if verbose:
            print(f"Found {len(channels)} unique channels: {channels}")
            print(f"Total states: {states}")
        
        # Build transition matrix
        transition_probs = _build_transition_matrix(
            journey_data, states, state_to_idx, min_journey_length, verbose
        )
        
        if verbose:
            print("\nTransition matrix:")
            for i, row in enumerate(transition_probs):
                print(f"{states[i]:<15} -> {dict(zip(states, [f'{x:.4f}' for x in row]))}")
        
        # Calculate removal effects
        removal_effects = _calculate_removal_effect(
            transition_probs, states, state_to_idx, max_iterations
        )
        
        # Normalize removal effects to sum to 1
        total_effect = sum(removal_effects.values())
        if total_effect > 0:
            removal_effects = {k: v / total_effect for k, v in removal_effects.items()}
        
        # Count total conversions for scaling
        total_conversions = journey_data['conversion'].sum()
        
        # Calculate attributed conversions
        attribution = {
            channel: removal_effects.get(channel, 0) * total_conversions 
            for channel in channels
        }
        
        # Convert to DataFrame
        result = pd.DataFrame(
            attribution.items(), 
            columns=['channel', 'attributed_conversions']
        )
        
        # Sort by attributed conversions
        result = result.sort_values('attributed_conversions', ascending=False)
        
        if verbose:
            print("\nAttribution results:")
            print(result.to_string(index=False))
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"Error in markov_chain_attribution: {str(e)}\n{traceback.format_exc()}"
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)
        if journey_data.empty:
            return pd.DataFrame(columns=['channel', 'attributed_conversions'])
        
        # Filter out journeys that are too short
        valid_customers = journey_data.groupby('customer_id').filter(
            lambda x: len(x) >= min_journey_length
        )['customer_id'].unique()
        
        if len(valid_customers) == 0:
            # Fallback to last-touch if no journeys meet the minimum length
            return last_touch_attribution(journey_data)
        
        # Get converting customers with valid journeys
        converting_customers = journey_data[
            (journey_data['conversion'] == True) & 
            (journey_data['customer_id'].isin(valid_customers))
        ]['customer_id'].unique()
        
        if len(converting_customers) == 0:
            return pd.DataFrame(columns=['channel', 'attributed_conversions'])
        
        # Initialize states and transition matrix
        channels = journey_data['channel'].unique()
        states = ['(start)'] + list(channels) + ['(conversion)', '(null)']
        n_states = len(states)
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        # Initialize transition matrix with small constant for numerical stability
        transition_matrix = np.full((n_states, n_states), 1e-10)
        
        # Count transitions
        for customer_id in converting_customers:
            customer_journey = journey_data[
                journey_data['customer_id'] == customer_id
            ].sort_values('timestamp')
            
            # Get conversion events
            conversion_events = customer_journey[customer_journey['conversion']]
            
            for _, conv_event in conversion_events.iterrows():
                # Get touchpoints before conversion
                touchpoints = customer_journey[
                    customer_journey['timestamp'] <= conv_event['timestamp']
                ]
                
                # Skip if journey is too short after filtering
                if len(touchpoints) < min_journey_length:
                    continue
                
                # Add path with start and end states
                path = ['(start)'] + touchpoints['channel'].tolist() + ['(conversion)']
                
                # Count transitions
                for i in range(len(path) - 1):
                    from_idx = state_to_idx[path[i]]
                    to_idx = state_to_idx[path[i + 1]]
                    transition_matrix[from_idx, to_idx] += 1
        
        # Convert counts to probabilities with smoothing
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_probs = np.divide(
            transition_matrix, 
            row_sums,
            out=np.full_like(transition_matrix, 1.0/n_states),  # Uniform if row sums to 0
            where=row_sums > 0
        )
        
        # Power iteration to find steady-state distribution
        n = transition_probs.shape[0]
        pi = np.ones(n) / n  # Initial uniform distribution
        
        for _ in range(max_iterations):
            new_pi = pi @ transition_probs
            
            # Check for convergence
            if np.allclose(new_pi, pi, rtol=1e-6, atol=1e-6):
                break
                
            pi = new_pi
        
        # Calculate removal effects with improved state handling
        removal_effects = {}
        conv_state = '(conversion)'
        
        # Calculate baseline conversion probability
        baseline_conversion = pi[state_to_idx[conv_state]] if conv_state in state_to_idx else 0
        
        if baseline_conversion <= 0:
            # If no baseline conversion, fall back to equal attribution
            return pd.DataFrame({
                'channel': channels,
                'attributed_conversions': [1.0/len(channels)] * len(channels)
            })
        
        for channel in channels:
            if channel not in state_to_idx:
                continue
                
            try:
                # Create a mask to keep all states except the current channel
                keep_indices = [i for i in range(n_states) if states[i] not in [channel]]
                
                if not keep_indices:
                    removal_effects[channel] = 0.0
                    continue
                    
                # Create reduced transition matrix
                reduced_matrix = transition_probs[np.ix_(keep_indices, keep_indices)]
                
                # Find the new conversion index in the reduced state space
                try:
                    reduced_conv_idx = [states[i] for i in keep_indices].index(conv_state)
                except ValueError:
                    # If conversion state was removed, set effect to baseline
                    removal_effects[channel] = baseline_conversion
                    continue
                
                # Renormalize the transition matrix
                row_sums = reduced_matrix.sum(axis=1, keepdims=True)
                reduced_matrix = np.divide(
                    reduced_matrix, 
                    row_sums,
                    out=np.full_like(reduced_matrix, 1.0/len(keep_indices)),
                    where=row_sums > 0
                )
                
                # Calculate steady-state distribution for reduced system
                n_reduced = len(keep_indices)
                reduced_pi = np.ones(n_reduced) / n_reduced
                
                # Power iteration with convergence check
                converged = False
                for _ in range(max_iterations):
                    new_pi = reduced_pi @ reduced_matrix
                    
                    # Check for convergence
                    if np.allclose(new_pi, reduced_pi, rtol=1e-6, atol=1e-10):
                        converged = True
                        break
                        
                    reduced_pi = new_pi
                
                if not converged:
                    print(f"Warning: Power iteration did not converge for channel {channel}")
                
                # Calculate removal effect
                reduced_conversion = reduced_pi[reduced_conv_idx] if n_reduced > 0 else 0
                removal_effect = max(0, baseline_conversion - reduced_conversion)
                removal_effects[channel] = removal_effect
                
            except Exception as e:
                print(f"Error processing channel {channel}: {str(e)}")
                removal_effects[channel] = 0.0
                continue
        
        # Convert to attribution
        total_effect = sum(removal_effects.values())
        total_conversions = len(converting_customers)
        
        if total_effect > 0:
            attribution = [
                {'channel': ch, 'attributed_conversions': (eff / total_effect) * total_conversions}
                for ch, eff in removal_effects.items() if eff > 0
            ]
        else:
            # Fallback to equal attribution if no removal effect
            valid_channels = [ch for ch in channels if ch in removal_effects]
            n_channels = len(valid_channels)
            attribution = [
                {'channel': ch, 'attributed_conversions': total_conversions / n_channels if n_channels > 0 else 0}
                for ch in valid_channels
            ]
        
        return pd.DataFrame(attribution) if attribution else pd.DataFrame(columns=['channel', 'attributed_conversions'])
        
    except Exception as e:
        import traceback
        print(f"Error in markov_chain_attribution: {str(e)}\n{traceback.format_exc()}")
        # Fallback to last-touch attribution on error
        return last_touch_attribution(journey_data)

def shapley_value_attribution(journey_data, max_channels=15, max_samples=1000, random_state=42):
    """
    Implements Shapley Value attribution model with performance optimizations.
    
    Uses a sampling-based approach to approximate Shapley values, which provides
    a fair distribution of credit across marketing channels based on their marginal
    contributions to conversions.
    
    Args:
        journey_data (pd.DataFrame): Customer journey data with columns:
                                  customer_id, timestamp, channel, conversion
        max_channels (int): Maximum number of channels to include in analysis.
                          If more channels exist, uses the most frequent ones.
        max_samples (int): Maximum number of samples for Shapley value approximation.
                         Higher values give more accurate results but are slower.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Attribution results with columns: channel, attributed_conversions
        
    Notes:
        - Uses sampling to handle the combinatorial explosion of Shapley values
        - Implements early stopping for faster convergence
        - Falls back to simpler models when data is insufficient
        - Handles both binary and continuous conversion values
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import train_test_split
        
        np.random.seed(random_state)
        
        if journey_data.empty:
            return pd.DataFrame(columns=['channel', 'attributed_conversions'])
        
        # Get top channels by frequency to limit complexity
        channel_counts = journey_data['channel'].value_counts()
        if len(channel_counts) > max_channels:
            top_channels = channel_counts.nlargest(max_channels).index.tolist()
            journey_data = journey_data[journey_data['channel'].isin(top_channels)]
        
        # Get converting and non-converting customers
        converting_customers = journey_data[
            journey_data['conversion'] == True
        ]['customer_id'].unique()
        
        if len(converting_customers) == 0:
            return pd.DataFrame(columns=['channel', 'attributed_conversions'])
            
        all_customers = journey_data['customer_id'].unique()
        non_converting_customers = np.setdiff1d(all_customers, converting_customers)
        
        # Sample non-converting customers if needed for balanced classes
        max_non_conv = min(len(converting_customers) * 5, len(non_converting_customers))
        if len(non_converting_customers) > max_non_conv:
            non_converting_customers = np.random.choice(
                non_converting_customers, max_non_conv, replace=False
            )
        
        # Prepare feature matrix
        channels = journey_data['channel'].unique()
        channel_to_idx = {channel: i for i, channel in enumerate(channels)}
        n_channels = len(channels)
        
        # Create features: binary indicators for channel presence
        X = np.zeros((len(all_customers), n_channels), dtype=int)
        y = np.zeros(len(all_customers), dtype=int)
        
        # Fill feature matrix
        for i, customer_id in enumerate(all_customers):
            customer_channels = journey_data[
                journey_data['customer_id'] == customer_id
            ]['channel'].unique()
            
            for ch in customer_channels:
                if ch in channel_to_idx:
                    X[i, channel_to_idx[ch]] = 1
            
            y[i] = 1 if customer_id in converting_customers else 0
        
        # Check if we have enough data for meaningful analysis
        if np.sum(y) < 5 or n_channels < 2:
            return last_touch_attribution(journey_data)
        
        # Split data for model validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state, stratify=y
        )
        
        # Train model with regularization to prevent overfitting
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
        )
        
        try:
            model.fit(X_train, y_train)
            
            # Check model quality
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            if test_score < 0.6 and abs(train_score - test_score) > 0.2:
                # Model is not generalizing well, fall back to simpler model
                return linear_attribution(journey_data)
                
        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return last_touch_attribution(journey_data)
        
        # Calculate Shapley values using sampling
        shapley_values = {channel: 0.0 for channel in channels}
        
        # Process customers in batches to manage memory
        batch_size = min(100, len(converting_customers))
        n_batches = (len(converting_customers) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(converting_customers))
            batch_customers = converting_customers[batch_start:batch_end]
            
            for customer_id in batch_customers:
                # Get customer's channels
                customer_channels = set(journey_data[
                    journey_data['customer_id'] == customer_id
                ]['channel'].unique())
                
                # Only consider channels that exist in our model
                customer_channels = [ch for ch in customer_channels if ch in channel_to_idx]
                
                if not customer_channels:
                    continue
                
                # Sample coalitions
                for _ in range(max_samples // len(customer_channels) + 1):
                    for channel in customer_channels:
                        # Random permutation of other channels
                        other_channels = [ch for ch in customer_channels if ch != channel]
                        np.random.shuffle(other_channels)
                        
                        # Find position of current channel in permutation
                        pos = np.random.randint(0, len(other_channels) + 1)
                        coalition = set(other_channels[:pos])
                        
                        # Calculate marginal contribution
                        # With channel
                        x_with = np.zeros(n_channels, dtype=int)
                        for ch in coalition | {channel}:
                            x_with[channel_to_idx[ch]] = 1
                        
                        # Without channel
                        x_without = np.zeros(n_channels, dtype=int)
                        for ch in coalition:
                            x_without[channel_to_idx[ch]] = 1
                        
                        # Predict probabilities
                        prob_with = model.predict_proba([x_with])[0][1]
                        prob_without = model.predict_proba([x_without])[0][1]
                        
                        # Update Shapley value for this channel
                        shapley_values[channel] += (prob_with - prob_without)
        
        # Normalize Shapley values to sum to total conversions
        total_shapley = sum(shapley_values.values())
        total_conversions = len(converting_customers)
        
        if total_shapley > 0:
            attribution = [
                {'channel': ch, 'attributed_conversions': max(0, (val / total_shapley) * total_conversions)}
                for ch, val in shapley_values.items()
            ]
        else:
            # Fallback to equal attribution if Shapley values sum to zero
            n_channels = len([ch for ch in channels if ch in shapley_values])
            if n_channels > 0:
                attribution = [
                    {'channel': ch, 'attributed_conversions': total_conversions / n_channels}
                    for ch in channels if ch in shapley_values
                ]
            else:
                return last_touch_attribution(journey_data)
        
        return pd.DataFrame(attribution)
        
    except Exception as e:
        import traceback
        print(f"Error in shapley_value_attribution: {str(e)}\n{traceback.format_exc()}")
        # Fallback to last-touch attribution on error
        return last_touch_attribution(journey_data)
