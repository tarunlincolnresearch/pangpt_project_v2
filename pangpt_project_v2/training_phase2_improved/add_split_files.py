#!/usr/bin/env python3
"""
Modify James's code to accept separate train/val/test files
"""
import re

with open('train_improved.py', 'r') as f:
    content = f.read()

print("Adding separate train/val/test file arguments...")

# 1. Replace single input_file with three separate arguments
old_arg = r'parser\.add_argument\("--input_file", type=str, required=True, help="Path to the input file"\)'
new_args = '''parser.add_argument("--train_file", type=str, required=True, help="Path to training windows file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation windows file")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test windows file")'''

content = re.sub(old_arg, new_args, content)
print("✓ Replaced --input_file with --train_file, --val_file, --test_file")

# 2. Update the variable assignment
content = re.sub(
    r'input_file = args\.input_file',
    '''train_file = args.train_file
val_file = args.val_file
test_file = args.test_file''',
    content
)
print("✓ Updated variable assignments")

# 3. Find and replace the data loading section
# We need to load from separate files instead of splitting
old_loading = r'genomes = load_genomes\(input_file\)'
new_loading = '''# Load pre-split data (genome-level split already done)
print(f"Loading pre-split data...")
print(f"  Train: {train_file}")
print(f"  Val:   {val_file}")
print(f"  Test:  {test_file}")

train_genomes = load_genomes(train_file)
val_genomes = load_genomes(val_file)
test_genomes = load_genomes(test_file)

print(f"Loaded {len(train_genomes)} train genomes")
print(f"Loaded {len(val_genomes)} val genomes")
print(f"Loaded {len(test_genomes)} test genomes")'''

content = re.sub(old_loading, new_loading, content)
print("✓ Updated data loading to use pre-split files")

# 4. Remove the internal splitting code (lines 462-466)
# Comment out the train_test_split calls
content = re.sub(
    r'([ ]+)train_genomes, val_genomes = train_test_split\(genomes.*?\n',
    r'\1# DISABLED: Using pre-split files instead\n\1# train_genomes, val_genomes = train_test_split(genomes...\n',
    content
)

content = re.sub(
    r'([ ]+)train_genomes, temp_genomes = train_test_split\(genomes.*?\n',
    r'\1# DISABLED: Using pre-split files instead\n\1# train_genomes, temp_genomes = train_test_split(genomes...\n',
    content
)

content = re.sub(
    r'([ ]+)val_genomes, test_genomes = train_test_split\(temp_genomes.*?\n',
    r'\1# val_genomes, test_genomes = train_test_split(temp_genomes...\n',
    content
)

print("✓ Disabled internal train_test_split (using pre-split files)")

# Write back
with open('train_improved.py', 'w') as f:
    f.write(content)

print("\n✅ Successfully modified to use pre-split files!")
print("\nNow the code will:")
print("  1. Accept --train_file, --val_file, --test_file")
print("  2. Load each file separately (no internal splitting)")
print("  3. Use your genome-level splits directly")
