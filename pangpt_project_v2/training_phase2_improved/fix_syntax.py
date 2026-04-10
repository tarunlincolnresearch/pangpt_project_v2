#!/usr/bin/env python3
import re

with open('train_improved.py', 'r') as f:
    content = f.read()

# Fix the empty else block
content = re.sub(
    r'(else:\n)(    # DISABLED: Using pre-split files instead\n    # train_genomes, temp_genomes)',
    r'\1    pass  # Using pre-split files loaded above\n\2',
    content
)

with open('train_improved.py', 'w') as f:
    f.write(content)

print("✓ Fixed indentation error")
