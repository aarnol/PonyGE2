import os 
import sys
import pandas as pd
grammar_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'grammars'))
sys.path.insert(0, grammar_dir)
print("Added grammar directory to sys.path:", grammar_dir)
from eeg_grammar_gen import generate_grammar_from_dataframe

train_path = './datasets/eeg/train.csv'
test_path = './datasets/eeg/test.csv'
grammar_path = './grammars/eeg_glm.bnf'
parameters_path = './parameters/glm_parameters.py'

grammar = generate_grammar_from_dataframe(pd.read_csv(train_path))
with open(grammar_path, 'w') as f:
    f.write(grammar)

#now run the script
import subprocess
print(os.getcwd())
# Run PonyGE2 with appropriate parameters
os.chdir('./src')
cmd = [
    'python', 'ponyge.py',
    '--parameters', 'glm_params.txt',
]

# Execute the command
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
#save output to a file
with open('../results/ponyge_output.txt', 'w') as f:
    f.write(result.stdout)

if result.stderr:
    print("Errors:", result.stderr)