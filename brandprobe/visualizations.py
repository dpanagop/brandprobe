import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sentence_transformers import SentenceTransformer


def plot_radar(df: pd.DataFrame, target_names: list[str], axis_col: str, 
               score: str = 'Sentiment', save_path: str | None = None) -> None:
    """
    Creates a dynamic radar chart comparing different targets across N categories.
    Optimized for sentiment scores ranging from -1 to 1.
    """
    # 1. Filter and group the data
    plot_df = df[df['Target'].isin(target_names)]
    
    # Calculate mean score for each target per axis category
    grouped = plot_df.groupby(['Target', axis_col])[score].mean().unstack()
    
    if grouped.empty:
        print(f"No data available for targets {target_names} and axis '{axis_col}'.")
        return
        
    categories = grouped.columns.tolist()   
    N = len(categories)
    
    # Check if we have enough points for a radar "shape"
    if N < 3:
        print(f"Warning: Radar charts require at least 3 categories to form a shape. "
              f"Detected only {N} ('{', '.join(categories)}'). Consider using a bar chart instead.")
    
    # 2. Setup Angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    # 3. Initialize the Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw axis lines and labels
    plt.xticks(angles[:-1], categories, color='black', size=11)
    
    # 4. Handle the -1 to 1 Sentiment Range
    # We set the limit slightly beyond -1/1 for visual breathing room
    ax.set_ylim(-1.1, 1.1) 
    
    # Draw circular grid lines at specific intervals
    grid_values = [-1.0, -0.5, 0, 0.5, 1.0]
    ax.set_rlabel_position(0)
    plt.yticks(grid_values, [str(v) for v in grid_values], color="grey", size=8)
    
    # Add a bold "Neutral/Zero" line to help distinguish positive from negative
    ax.plot(np.linspace(0, 2*np.pi, 100), np.zeros(100), color='black', 
            linestyle='--', linewidth=1, alpha=0.6, label="_nolegend_")
    
    # 5. Plot each Target
    for target in target_names:
        if target in grouped.index:
            values = grouped.loc[target].values.tolist()
            # Handle potential NaNs in the pivot table
            if any(pd.isna(values)):
                print(f"Warning: Target '{target}' has missing data for some categories. Result may look broken.")
            
            values += values[:1] # Close the loop
            
            line, = ax.plot(angles, values, linewidth=2, linestyle='solid', label=target)
            ax.fill(angles, values, alpha=0.15)
            
    plt.title(f'Comparison of Targets across {axis_col}\n(Sentiment Scale: -1 to 1)', size=14, pad=30)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # 6. Save or Show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_semantic_map(df: pd.DataFrame, target_filter: str, color_by: str, save_path: str | None = None) -> None:
    """
    Creates a 2D UMAP semantic map of generated responses, colored by a specific attribute.
    
    Args:
        df: The DataFrame containing audit results with a 'Response' column.
        target_filter: The 'Target' value to filter by (e.g., 'OpenAI').
        color_by: The column used to color the scattered points (e.g., 'Persona', 'Methodology').
        save_path: Optional path to save the generated plot. If None, plt.show() is called.
    """
    # Filter the dataframe
    plot_df = df[df['Target'] == target_filter].copy()
    
    if plot_df.empty:
        print(f"No data found for Target: '{target_filter}'.")
        return
        
    if 'Response' not in plot_df.columns:
        raise ValueError("DataFrame must contain a 'Response' column for semantic mapping.")
        
    # Drop rows with NaN responses
    plot_df = plot_df.dropna(subset=['Response', color_by])
    
    if plot_df.empty:
         print("No valid 'Response' data available after filtering.")
         return

    responses = plot_df['Response'].tolist()
    
    # Load model and generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(responses)
    
    # Reduce dimensionality using UMAP
    reducer = umap.UMAP(n_neighbors=min(15, len(embeddings)-1), min_dist=0.1, n_components=2, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Add 2D coordinates back to dataframe for plotting
    plot_df['UMAP_X'] = embedding_2d[:, 0]
    plot_df['UMAP_Y'] = embedding_2d[:, 1]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='UMAP_X', 
        y='UMAP_Y',
        hue=color_by,
        palette='viridis',
        alpha=0.7,
        s=100
    )
    
    plt.title(f'UMAP Semantic Map for {target_filter} (Colored by {color_by})', size=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_skew_comparison(skew_df: pd.DataFrame, x_col: str, y_col: str = 'Skewness', hue_col: str | None = None, save_path: str | None = None) -> None:
    """
    Creates a bar plot comparing the skewness across different groups.
    
    Args:
        skew_df: The DataFrame returned by calculate_sentiment_skew(df, group_cols=[...]).
        x_col: The primary category to plot on the x-axis (e.g., 'Target' or 'Methodology').
        y_col: The column containing skewness values (default 'Skewness').
        hue_col: Optional secondary grouping to color the bars by (e.g., 'Persona').
        save_path: Optional path to save the generated plot. If None, plt.show() is called.
    """
    
    if x_col not in skew_df.columns:
        print(f"Error: x_col '{x_col}' not found in DataFrame.")
        return
        
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=skew_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette='magma' if hue_col else 'viridis'
    )
    
    # Add horizontal lines at -0.5 and 0.5 to show the interpretation thresholds
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Right Skewed / Leans Negative (> 0.5)')
    plt.axhline(-0.5, color='green', linestyle='--', alpha=0.5, label='Left Skewed / Leans Positive (< -0.5)')
    plt.axhline(0, color='gray', linestyle='-', alpha=0.3) # Center line
    
    title = f'Sentiment Skew Comparison by {x_col}'
    if hue_col:
        title += f' (Grouped by {hue_col})'
        
    plt.title(title, size=14)
    plt.ylabel('Fisher-Pearson Skewness')
    
    # Adjust legend position to avoid blocking bars
    if hue_col:
        plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
