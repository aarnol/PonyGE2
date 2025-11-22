import pandas as pd
import numpy as np
from sklearn.datasets import make_regression

# Set random seed for reproducibility
np.random.seed(42)

# Generate the test DataFrame
import os
print(f"Current working directory: {os.getcwd()}")
test_df = pd.read_csv('./datasets/eeg/train.csv')
#add target column of random values to test_df
test_df['target'] = np.random.randn(test_df.shape[0])


# Now let's test our grammar generation functions
def generate_grammar_from_dataframe(df, target_col='target'):
    """Generate BNF grammar string from DataFrame columns"""
    
    # Get column names (excluding target variable)
    features = [col for col in df.columns if col != target_col]
    
    # Base grammar structure
    grammar_lines = [
        "<expr> ::= <expr><op><expr>|sin(<expr>)|cos(<expr>)|exp(<expr>)|pow(<expr>, 2)|sqrt(<expr>)|<var>",
        "<op> ::= +|-|/|*"
    ]
    
    # Generate variable rule with all feature names
    var_rule = "<var> ::= " + "|".join(features)
    grammar_lines.append(var_rule)
    
   
    
    return "\n".join(grammar_lines)

def generate_typed_grammar(df, target_col='target'):
    """Generate grammar with feature type groupings"""
    
    # Remove target column
    feature_df = df.drop(columns=[target_col])
    
    # Separate numeric and categorical features
    numeric_features = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    grammar_lines = [
        "<expr> ::= <expr><op><expr>|<func>(<expr>)|<var>|<const>",
        "<func> ::= sin|cos|exp|sqrt|log|abs",
        "<op> ::= +|-|/|*",
        "<var> ::= <numeric_var>"
    ]
    
    # Add categorical variables to var rule if they exist
    if categorical_features:
        grammar_lines[3] += "|<categorical_var>"
    
    # Add numeric variables rule
    if numeric_features:
        numeric_rule = "<numeric_var> ::= " + "|".join(numeric_features)
        grammar_lines.append(numeric_rule)
    
    # Add categorical variables rule (these would need encoding in practice)
    if categorical_features:
        categorical_rule = "<categorical_var> ::= " + "|".join(categorical_features)
        grammar_lines.append(categorical_rule)
    
    # Add constants
    grammar_lines.append("<const> ::= 0.1|1.0|2.0|5.0|10.0")
    
    return "\n".join(grammar_lines)
if __name__ == "__main__":
    # Test the grammar generation functions
    print("\n" + "="*60)
    print("TESTING GRAMMAR GENERATION")
    print("="*60)

    print("\n1. Basic Grammar Generation:")
    basic_grammar = generate_grammar_from_dataframe(test_df)
    print(basic_grammar)

    print("\n2. Typed Grammar Generation:")
    typed_grammar = generate_typed_grammar(test_df)
    print(typed_grammar)

    # Save grammars to files
    with open('basic_grammar.pybnf', 'w') as f:
        f.write(basic_grammar)

    with open('typed_grammar.pybnf', 'w') as f:
        f.write(typed_grammar)

    print("\nGrammar files saved:")
    print("- basic_grammar.pybnf")
    print("- typed_grammar.pybnf")

    # Show feature counts
    numeric_features = test_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = test_df.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"\nFeature Summary:")
    print(f"- Total features: {len(test_df.columns) - 1}")  # -1 for target
    print(f"- Numeric features: {len(numeric_features)} {numeric_features}")
    print(f"- Categorical features: {len(categorical_features)} {categorical_features}")