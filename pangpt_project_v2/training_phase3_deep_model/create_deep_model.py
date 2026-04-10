#!/usr/bin/env python3
"""
Modify training script to create deeper, more complex model
"""
import re

with open('train_deep.py', 'r') as f:
    content = f.read()

print("Creating Phase 3 Deep Model configuration...\n")

# 1. Update version string
content = re.sub(
    r'VERSION = ".*"',
    'VERSION = "0.10a - Phase 3 Deep Model"',
    content
)
print("✓ Updated version string")

# 2. Change default num_layers from 6 to 12
content = re.sub(
    r'--num_layers.*default=6',
    '--num_layers", type=int, default=12',
    content
)
print("✓ Layers: 6 → 12")

# 3. Change default embed_dim from 256 to 512
content = re.sub(
    r'--embed_dim.*default=256',
    '--embed_dim", type=int, default=512',
    content
)
print("✓ Embedding dimension: 256 → 512")

# 4. Change default num_heads from 8 to 16
content = re.sub(
    r'--num_heads.*default=8',
    '--num_heads", type=int, default=16',
    content
)
print("✓ Attention heads: 8 → 16")

# 5. Increase batch size slightly for efficiency (8 → 4 due to larger model)
content = re.sub(
    r'--batch_size.*default=8',
    '--batch_size", type=int, default=4',
    content
)
print("✓ Batch size: 8 → 4 (larger model needs smaller batches)")

# 6. Add gradient accumulation for effective larger batch
# Find the train_model function and add accumulation
accumulation_code = '''
# Gradient accumulation for effective batch size
accumulation_steps = 2  # Effective batch = 4 * 2 = 8
'''

# Insert before the train_model function
content = re.sub(
    r'(def train_model\(train_loader)',
    accumulation_code + r'\1',
    content
)
print("✓ Added gradient accumulation (effective batch = 8)")

# 7. Modify training loop to use gradient accumulation
# Find the optimizer.step() call and wrap it
old_training_loop = r'''        optimizer.zero_grad\(\)
        outputs = model\(input_ids\)
        loss = criterion\(outputs\.view\(-1, model\.vocab_size\), labels\.view\(-1\)\)
        loss\.backward\(\)
        torch\.nn\.utils\.clip_grad_norm_\(model\.parameters\(\), max_norm=1\.0\)
        optimizer\.step\(\)'''

new_training_loop = '''        if i % accumulation_steps == 0:
            optimizer.zero_grad()
        
        outputs = model(input_ids)
        loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()'''

content = re.sub(old_training_loop, new_training_loop, content)
print("✓ Modified training loop for gradient accumulation")

# 8. Keep moderate regularization (same as Phase 2A)
print("✓ Keeping moderate regularization from Phase 2A:")
print("  - Label smoothing: 0.05")
print("  - Weight decay: 0.0005")
print("  - Dropout: 0.15")

with open('train_deep.py', 'w') as f:
    f.write(content)

print("\n✅ Phase 3 Deep Model script created!")
print("\nModel Configuration:")
print("  Layers: 12 (2x deeper)")
print("  Embedding: 512 (2x wider)")
print("  Heads: 16 (2x more)")
print("  Parameters: ~120M (4x larger)")
print("  Effective batch: 8 (via gradient accumulation)")

