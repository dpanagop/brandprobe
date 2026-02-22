import pandas as pd
import numpy as np
import os
from brandprobe.analytics import get_consistency_metrics, calculate_sentiment_skew
from brandprobe.visualizations import plot_radar, plot_semantic_map, plot_skew_comparison
def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic BrandProbe audit results for testing."""
    targets = ['OpenAI', 'AzureOpenAI']
    personas = ['Friendly', 'Professional', 'Skeptical']
    methodologies = ['Direct', 'Indirect', 'Hypothetical']
    test_cases = [f'Test_{i}' for i in range(1, 6)]
    
    data = []
    for t in targets:
        for p in personas:
            for m in methodologies:
                for tc in test_cases:
                    # Synthetic sentiment score
                    score = np.random.uniform(0.3, 0.9)
                    
                    # Synthetic responses with some semantic relation
                    if p == 'Friendly':
                        response = f"I think {t} is great! It works {m} perfectly."
                    elif p == 'Professional':
                        response = f"{t} exhibits strong capabilities in {m} contexts."
                    else:
                        response = f"I'm not sure if {t} is reliable when tested {m}."
                        
                    data.append({
                        'Target': t,
                        'Persona': p,
                        'Methodology': m,
                        'Test Case': tc,
                        'Score': score,
                        'Response': response
                    })
                    
    return pd.DataFrame(data)

def test_visuals():
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    print("\n--- Testing Analytics ---")
    
    # 1. Test Semantic Consistency
    print("Calculating Semantic Consistency...")
    consistency_df = get_consistency_metrics(df, group_cols=['Target', 'Persona'])
    print(consistency_df.head())
    
    # 2. Test Sentiment Skew
    print("\nCalculating Global Sentiment Skew...")
    global_skew = calculate_sentiment_skew(df)
    print(global_skew)

    print("\nCalculating Grouped Sentiment Skew (by Target and Persona)...")
    skew_df = calculate_sentiment_skew(df, group_cols=['Target', 'Persona'])
    print(skew_df.head(6))
    
    print("\n--- Testing Visualizations ---")
    
    # Optional output directory for test images
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Test Radar Chart
    radar_path = os.path.join(output_dir, "radar_chart.png")
    print(f"Generating Radar Chart -> {radar_path}")
    plot_radar(
        df=df,
        target_names=['OpenAI', 'AzureOpenAI'],
        axis_col='Methodology',
        save_path=radar_path
    )
    
    # 2. Test UMAP Semantic Map
    umap_path = os.path.join(output_dir, "umap_map.png")
    print(f"Generating UMAP Map -> {umap_path}")
    plot_semantic_map(
        df=df,
        target_filter='OpenAI',
        color_by='Persona',
        save_path=umap_path
    )
    
    # 3. Test Skew Comparison Plot
    skew_path = os.path.join(output_dir, "skew_comparison.png")
    print(f"Generating Skew Comparison Plot -> {skew_path}")
    plot_skew_comparison(
        skew_df=skew_df,
        x_col='Target',
        hue_col='Persona',
        save_path=skew_path
    )
    
    print("\nVerification Complete! Check the 'test_output' directory for the charts.")

if __name__ == "__main__":
    test_visuals()
