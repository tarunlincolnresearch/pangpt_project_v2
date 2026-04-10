#!/usr/bin/env python3
import re

with open('train_improved.py', 'r') as f:
    content = f.read()

print("Fixing input_file references...")

# Replace input_file checks with train_file checks
content = re.sub(
    r'if not os\.path\.isfile\(input_file\):',
    'if not os.path.isfile(train_file):',
    content
)

content = re.sub(
    r'print\(f"Error: Input file \{input_file\} not found\."\)',
    'print(f"Error: Train file {train_file} not found.")',
    content
)

# Also check for any other input_file references
content = re.sub(
    r'\binput_file\b',
    'train_file',
    content
)

# But restore the argument name
content = re.sub(
    r'--train_file", type=str, required=True, help="Path to training windows file"',
    '--train_file", type=str, required=True, help="Path to training windows file"',
    content
)

with open('train_improved.py', 'w') as f:
    f.write(content)

print("✓ Fixed all input_file references")
