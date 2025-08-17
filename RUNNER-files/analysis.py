import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

### HELPER FUNCTIONS FOR COMPLEXITY & ENTROPY ###

def calculate_lz_complexity(input_string):
    """Calculates the Lempel-Ziv complexity of a binary string."""
    if not isinstance(input_string, str):
        input_string = str(input_string)
    
    dictionary = set()
    i = 0
    complexity = 0
    while i < len(input_string):
        substring = ""
        while i < len(input_string):
            substring += input_string[i]
            if substring not in dictionary:
                dictionary.add(substring)
                complexity += 1
                break
            i += 1
        i += 1
    return complexity

def calculate_shannon_entropy(input_string):
    """Calculates the Shannon entropy of a string."""
    if not isinstance(input_string, str):
        input_string = str(input_string)
    
    prob = [float(input_string.count(c)) / len(input_string) for c in dict.fromkeys(list(input_string))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def calculate_approximate_entropy(data, m=2, r_multiplier=0.2):
    """
    Calculates the Approximate Entropy (ApEn) of a time series.
    
    Args:
        data (list or np.array): A 1D time series of numbers.
        m (int): The embedding dimension (length of sequences to compare).
        r_multiplier (float): The tolerance factor, multiplied by the standard deviation of the data.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
        
    if data.ndim != 1 or len(data) < 20: # ApEn is unreliable for very short series
        return np.nan 

    N = len(data)
    r = r_multiplier * np.std(data)

    def _phi(m_val):
        x = np.array([data[i:i+m_val] for i in range(N - m_val + 1)])
        C = np.sum(np.max(np.abs(x[:, np.newaxis] - x), axis=2) <= r, axis=0) / (N - m_val + 1.0)
        return np.sum(np.log(C)) / (N - m_val + 1.0)

    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)
    
    return phi_m - phi_m_plus_1

### DATA PROCESSING & ANALYSIS FUNCTIONS ###

def process_pattern_matching_data(filepath="results_pattern_matching.csv"):
    """Loads, processes, and analyzes the QHNN pattern matching data."""
    print("="*60)
    print("🔬 ANALYSIS: Pattern Matching (QHNN)")
    print("="*60)
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run runner.py first.\n")
        return

    # 1. Calculate Predictor Variables
    df['lempel_ziv_complexity'] = df['raw_output_bitstring'].apply(calculate_lz_complexity)
    df['kolmogorov_approx'] = df['lempel_ziv_complexity'] # As per dissertation methodology [cite: 1602]
    df['shannon_entropy'] = df['raw_output_bitstring'].apply(calculate_shannon_entropy)
    df['approximate_entropy'] = np.nan # Not applicable to single-string output

    # 2. Calculate Dependent Variable (Emergence Score)
    # [cite_start]PM-ES = ω1*Recall Fidelity Gain + ω2*LZC_output + ω3*Η(χ) [cite: 1546]
    # Assuming equal weights for this example (ω1=ω2=ω3=1/3)
    # Using 'prediction_accuracy' as a proxy for 'Recall Fidelity Gain'
    df['emergence_score'] = (1/3) * df['prediction_accuracy'] + \
                            (1/3) * df['lempel_ziv_complexity'] + \
                            (1/3) * df['shannon_entropy']
    
    # Normalize the score to be between 0 and 1 for easier interpretation
    df['emergence_score'] = (df['emergence_score'] - df['emergence_score'].min()) / \
                            (df['emergence_score'].max() - df['emergence_score'].min())

    # 3. Perform Statistical Analysis
    run_statistical_analysis(df, 'Pattern Matching')

def process_problem_solving_data(filepath="results_problem_solving.csv"):
    """Loads, processes, and analyzes the VQNN problem solving data."""
    print("\n" + "="*60)
    print("🔬 ANALYSIS: Adaptive Problem Solving (VQNN)")
    print("="*60)
    
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run runner.py first.\n")
        return
        
    # Clean up data: convert path string to list of tuples, handle failed runs
    df = df[df['steps_to_goal'] != -1].copy() # Exclude runs that didn't find the goal
    df['solution_path_list'] = df['solution_path'].apply(eval)

    # 1. Calculate Predictor Variables
    df['lempel_ziv_complexity'] = df['solution_path'].apply(calculate_lz_complexity)
    df['kolmogorov_approx'] = df['lempel_ziv_complexity']
    df['shannon_entropy'] = df['solution_path'].apply(calculate_shannon_entropy)
    # For ApEn, convert 2D path to 1D time series of Manhattan distance from start
    df['approximate_entropy'] = df['solution_path_list'].apply(
        lambda path: calculate_approximate_entropy([abs(p[0]-path[0][0]) + abs(p[1]-path[0][1]) for p in path])
    )
    
    # 2. Calculate Dependent Variable (Emergence Score)
    # [cite_start]PS-ES = z1*Performance Discontinuity + z2*ApEn + z3*LZC_solution [cite: 1552]
    # Using '1 / steps_to_goal' as proxy for 'Performance Discontinuity'
    # Assuming equal weights (z1=z2=z3=1/3)
    df['performance_proxy'] = 1 / df['steps_to_goal']
    df['emergence_score'] = (1/3) * df['performance_proxy'] + \
                            (1/3) * df['approximate_entropy'].fillna(0) + \
                            (1/3) * df['lempel_ziv_complexity']
    
    df['emergence_score'] = (df['emergence_score'] - df['emergence_score'].min()) / \
                            (df['emergence_score'].max() - df['emergence_score'].min())
                            
    # 3. Perform Statistical Analysis
    run_statistical_analysis(df, 'Problem Solving')
    
def process_creative_thinking_data(filepath="results_creative_thinking.csv"):
    """Loads, processes, and analyzes the QAM creative thinking data."""
    print("\n" + "="*60)
    print("🔬 ANALYSIS: Creative Thinking (QAM)")
    print("="*60)
    
    try:
        df = pd.read_csv(filepath)
        if 'ai_rater_novelty_score' not in df.columns:
            print("Warning: 'ai_rater_novelty_score' column not found.")
            print("Please add it to the CSV with scores from 1-7 to run this analysis.\n")
            return
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Please run runner.py first.\n")
        return

    # 1. Calculate Predictor Variables
    df['lempel_ziv_complexity'] = df['raw_output_vector'].apply(calculate_lz_complexity)
    df['kolmogorov_approx'] = df['lempel_ziv_complexity']
    df['shannon_entropy'] = df['raw_output_vector'].apply(calculate_shannon_entropy)
    df['approximate_entropy'] = np.nan

    # 2. Calculate Dependent Variable (Emergence Score)
    # [cite_start]CT-ES = 1/3 * (Novelty Index + LZC_output + H(X)) [cite: 1557]
    # Using the 'ai_rater_novelty_score' as the 'Novelty Index'
    df['emergence_score'] = (1/3) * (df['ai_rater_novelty_score'] + \
                                     df['lempel_ziv_complexity'] + \
                                     df['shannon_entropy'])
    
    df['emergence_score'] = (df['emergence_score'] - df['emergence_score'].min()) / \
                            (df['emergence_score'].max() - df['emergence_score'].min())

    # 3. Perform Statistical Analysis
    run_statistical_analysis(df, 'Creative Thinking')

def run_statistical_analysis(df, task_name):
    """Performs and prints the core statistical analysis for a given dataframe."""
    
    predictors = ['kolmogorov_approx', 'lempel_ziv_complexity', 'shannon_entropy', 'approximate_entropy']
    # Drop ApEn if it's all NaN (for QHNN and QAM tasks)
    predictors = [p for p in predictors if not df[p].isnull().all()]
    
    df_clean = df[['emergence_score'] + predictors].dropna()

    if df_clean.empty:
        print("No valid data for statistical analysis after cleaning.")
        return

    X = df_clean[predictors]
    y = df_clean['emergence_score']

    # --- Descriptive Statistics ---
    print("\n[ 1. Descriptive Statistics ]\n")
    print(df_clean.describe())

    # --- Correlation Matrix ---
    print("\n[ 2. Correlation Matrix ]\n")
    corr_matrix = df_clean.corr()
    print(corr_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for {task_name}')
    plt.show()

    # --- Multiple Linear Regression ---
    print("\n[ 3. Multiple Linear Regression Results ]\n")
    X = sm.add_constant(X) # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    print(model.summary())
    
    print(f"\nCONCLUSION for {task_name}:")
    p_value = model.f_pvalue
    if p_value < 0.05:
        print(f"The model is statistically significant (p = {p_value:.4f} < 0.05).")
        print("We REJECT the null hypothesis (H1_0) and accept the alternative (H1_a).")
        print("There IS a statistically significant correlation between the measures and emergent behaviors.")
    else:
        print(f"The model is NOT statistically significant (p = {p_value:.4f} >= 0.05).")
        print("We FAIL TO REJECT the null hypothesis (H1_0).")
        print("There is NO statistically significant correlation detected.")


if __name__ == "__main__":
    process_pattern_matching_data()
    process_problem_solving_data()
    process_creative_thinking_data()