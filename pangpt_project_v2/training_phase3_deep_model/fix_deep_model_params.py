#!/usr/bin/env python3
import re

with open('train_deep.py', 'r') as f:
    content = f.read()

print("Fixing deep model parameters...\n")

# Fix num_layers
content = re.sub(
    r'parser\.add_argument\("--num_layers", type=int, default=\d+',
    'parser.add_argument("--num_layers", type=int, default=12',
    content
)
print("✓ Fixed num_layers → 12")

# Fix embed_dim
content = re.sub(
    r'parser\.add_argument\("--embed_dim", type=int, default=\d+',
    'parser.add_argument("--embed_dim", type=int, default=512',
    content
)
print("✓ Fixed embed_dim → 512")

# Fix num_heads
content = re.sub(
    r'parser\.add_argument\("--num_heads", type=int, default=\d+',
    'parser.add_argument("--num_heads", type=int, default=16',
    content
)
print("✓ Fixed num_heads → 16")

# Fix batch_size
content = re.sub(
    r'parser\.add_argument\("--batch_size", type=int, default=\d+',
    'parser.add_argument("--batch_size", type=int, default=4',
    content
)
print("✓ Fixed batch_size → 4")

with open('train_deep.py', 'w') as f:
    f.write(content)

print("\n✅ All parameters fixed!")

# Verify
with open('train_deep.py', 'r') as f:
    content = f.read()
    
layers = re.search(r'--num_layers.*?default=(\d+)', content)
embed = re.search(r'--embed_dim.*?default=(\d+)', content)
heads = re.search(r'--num_heads.*?default=(\d+)', content)
batch = re.search(r'--batch_size.*?default=(\d+)', content)

print("\nVerification:")
print(f"  Layers: {layers.group(1)}")
print(f"  Embed: {embed.group(1)}")
print(f"  Heads: {heads.group(1)}")
print(f"  Batch: {batch.group(1)}")

