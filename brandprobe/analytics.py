import pandas as pd
import numpy as np
from scipy.stats import skew
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings that might occur during embedding generation
warnings.filterwarnings("ignore", category=FutureWarning)

def get_consistency_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation of Cosine Similarity for generated responses 
    within specified groups, using local SentenceTransformer embeddings.

    Args:
        df: The DataFrame containing the audit results. Must have a 'Response' column.
        group_cols: A list of columns to group by (e.g., ['Target', 'Persona']).

    Returns:
        A DataFrame containing the mean and std dev of cosine similarities for each group.
    """
    if 'Response' not in df.columns:
        raise ValueError("DataFrame must contain a 'Response' column.")
    
    # Load the local model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    results = []
    
    # Group the dataframe
    for name, group in df.groupby(group_cols):
        responses = group['Response'].dropna().tolist()
        
        if len(responses) < 2:
            # Need at least 2 responses to calculate similarity
            results.append({
                **dict(zip(group_cols, name if isinstance(name, tuple) else (name,))),
                'Cosine_Sim_Mean': np.nan,
                'Cosine_Sim_Std': np.nan,
                'N_Samples': len(responses)
            })
            continue
            
        # Generate embeddings
        embeddings = model.encode(responses)
        
        # Calculate pairwise cosine similarities
        sim_matrix = cosine_similarity(embeddings)
        
        # Extract upper triangle (excluding diagonal) to get unique pairs
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        pairwise_similarities = sim_matrix[upper_tri_indices]
        
        results.append({
            **dict(zip(group_cols, name if isinstance(name, tuple) else (name,))),
            'Cosine_Sim_Mean': np.mean(pairwise_similarities),
            'Cosine_Sim_Std': np.std(pairwise_similarities),
            'N_Samples': len(responses)
        })
        
    return pd.DataFrame(results)


def _interpret_skew(skew_val: float) -> str:
    """Helper function to interpret the Fisher-Pearson skewness value."""
    if pd.isna(skew_val):
        return "Not Enough Data"
    if skew_val > 0.5:
        return "Right Skewed / Leans Negative"
    elif skew_val < -0.5:
        return "Left Skewed / Leans Positive"
    else:
        return "Relatively Symmetrical"

def calculate_sentiment_skew(df: pd.DataFrame, target_col: str = 'Score', group_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Calculates the Fisher-Pearson skewness of sentiment scores to identify
    if the model has a "Pollyanna" (overly positive) or negative bias.
    
    A negative skew indicates a long tail on the left (bulk of values are positive).
    A positive skew indicates a long tail on the right (bulk of values are negative).
    
    Args:
        df: The DataFrame containing audit results.
        target_col: The numerical column to calculate skewness on (default 'Score').
        group_cols: Optional list of columns to slice the skewness by (e.g. ['Target', 'Persona']).

    Returns:
        A pandas DataFrame showing the Skewness and its textual Interpretation for each group 
        (or globally if no group_cols provided).
    """
    
    def safe_skew(series):
        clean_scores = series.dropna().to_numpy()
        if len(clean_scores) < 3 or np.var(clean_scores) == 0:
            return np.nan
        return float(skew(clean_scores, bias=False))

    if group_cols:
        # Group by the specified columns and apply our safe_skew wrapper
        grouped_skew = df.groupby(group_cols)[target_col].apply(safe_skew).reset_index()
        grouped_skew.rename(columns={target_col: 'Skewness'}, inplace=True)
        # Add Interpretation
        grouped_skew['Interpretation'] = grouped_skew['Skewness'].apply(_interpret_skew)
        return grouped_skew
    else:
        # Calculate global skew if no grouping is requested
        global_skew = safe_skew(df[target_col])
        return pd.DataFrame([{
            'Global': 'All Data',
            'Skewness': global_skew,
            'Interpretation': _interpret_skew(global_skew)
        }])

