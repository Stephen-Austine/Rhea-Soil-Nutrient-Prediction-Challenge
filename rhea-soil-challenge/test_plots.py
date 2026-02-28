import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load training data
train = pd.read_csv("Train.csv")

print("=== Testing Plot Generation ===")

# Test 1: Geographic Distribution map
print("\n1. Testing Geographic Distribution Map")
try:
    sample_size = min(2000, len(train))
    train_sample = train.sample(sample_size, random_state=42)
    color_col = 'N' if 'N' in train_sample.columns else train_sample.columns[2]
    
    fig_map = px.scatter_map(
        train_sample,
        lat="Latitude", lon="Longitude", color=color_col,
        zoom=3, height=500, map_style="open-street-map",
        color_continuous_scale="Viridis",
        title=f"Spatial Distribution of {color_col} Concentration"
    )
    fig_map.update_layout(
        margin=dict(t=40, b=0, l=0, r=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    print(f"✓ Map created successfully with color column: {color_col}")
    print(f"  Sample size: {sample_size}")
    print(f"  Data points: {len(train_sample)}")
except Exception as e:
    print(f"✗ Error creating map: {e}")

# Test 2: pH vs N plot
print("\n2. Testing pH vs N Plot")
try:
    if 'ph' in train.columns and 'N' in train.columns:
        df_ph = train[['ph', 'N']].dropna()
        if len(df_ph) > 0:
            fig_ph = px.scatter(
                df_ph,
                x='ph',
                y='N',
                title=f'pH vs N',
                labels={'ph': 'pH Level', 'N': 'N (mg/kg)'},
                color_discrete_sequence=['#3b82f6'],
                opacity=0.6
            )
            fig_ph.update_layout(
                height=300,
                margin=dict(t=40, b=0, l=0, r=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            print(f"✓ pH vs N plot created successfully")
            print(f"  Data points: {len(df_ph)}")
        else:
            print("✗ No valid data points for pH vs N")
    else:
        print(f"✗ Columns not found: ph={'ph' in train.columns}, N={'N' in train.columns}")
except Exception as e:
    print(f"✗ Error creating pH vs N plot: {e}")

# Test 3: pH vs K plot
print("\n3. Testing pH vs K Plot")
try:
    if 'ph' in train.columns and 'K' in train.columns:
        df_ph = train[['ph', 'K']].dropna()
        if len(df_ph) > 0:
            fig_ph = px.scatter(
                df_ph,
                x='ph',
                y='K',
                title=f'pH vs K',
                labels={'ph': 'pH Level', 'K': 'K (mg/kg)'},
                color_discrete_sequence=['#3b82f6'],
                opacity=0.6
            )
            fig_ph.update_layout(
                height=300,
                margin=dict(t=40, b=0, l=0, r=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            print(f"✓ pH vs K plot created successfully")
            print(f"  Data points: {len(df_ph)}")
        else:
            print("✗ No valid data points for pH vs K")
    else:
        print(f"✗ Columns not found: ph={'ph' in train.columns}, K={'K' in train.columns}")
except Exception as e:
    print(f"✗ Error creating pH vs K plot: {e}")

# Test 4: Nutrient Relationships plots
print("\n4. Testing Nutrient Relationships Plots")
try:
    nutrient_pairs = [('K', 'Ca'), ('K', 'Mg'), ('Ca', 'Mg')]
    available_pairs = [(n1, n2) for n1, n2 in nutrient_pairs if n1 in train.columns and n2 in train.columns]
    
    if available_pairs:
        colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444']
        
        for i, (nutrient1, nutrient2) in enumerate(available_pairs):
            df_scatter = train[[nutrient1, nutrient2]].dropna()
            if len(df_scatter) > 0:
                fig_scatter = px.scatter(
                    df_scatter,
                    x=nutrient1,
                    y=nutrient2,
                    title=f'{nutrient1} vs {nutrient2}',
                    labels={nutrient1: f'{nutrient1} (mg/kg)', nutrient2: f'{nutrient2} (mg/kg)'},
                    color_discrete_sequence=[colors[i % len(colors)]],
                    opacity=0.6
                )
                fig_scatter.update_layout(
                    height=300,
                    margin=dict(t=40, b=0, l=0, r=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                print(f"✓ {nutrient1} vs {nutrient2} plot created successfully")
                print(f"  Data points: {len(df_scatter)}")
            else:
                print(f"✗ No valid data points for {nutrient1} vs {nutrient2}")
    else:
        print("✗ No available nutrient pairs")
except Exception as e:
    print(f"✗ Error creating nutrient relationships plots: {e}")
