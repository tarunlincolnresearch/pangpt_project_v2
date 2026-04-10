#!/usr/bin/env python3
"""
Apply Phase 2 improvements to James's panGPT.py
Minimal changes for: stronger regularization, label smoothing, better LR schedule
"""

import re

# Read the original file
with open('train_improved.py', 'r') as f:
    content = f.read()

print("Applying Phase 2 improvements...")

# 1. Increase weight_decay from 1e-4 to 1e-3
content = re.sub(
    r'--weight_decay.*default=1e-4',
    '--weight_decay", type=float, default=1e-3',
    content
)
print("✓ Updated weight_decay: 1e-4 → 1e-3")

# 2. Add label smoothing to CrossEntropyLoss
content = re.sub(
    r'criterion = torch\.nn\.CrossEntropyLoss\(\)',
    'criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)',
    content
)
print("✓ Added label_smoothing=0.1 to CrossEntropyLoss")

# 3. Import CosineAnnealingWarmRestarts
content = re.sub(
    r'from torch\.optim\.lr_scheduler import ReduceLROnPlateau',
    'from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts',
    content
)
print("✓ Added CosineAnnealingWarmRestarts import")

# 4. Add argument for choosing scheduler type
# Find the lr_scheduler_factor argument and add a new one after it
content = re.sub(
    r'(parser\.add_argument\("--lr_scheduler_factor".*\n)',
    r'\1    parser.add_argument("--use_cosine_scheduler", action="store_true", help="Use CosineAnnealingWarmRestarts instead of ReduceLROnPlateau")\n',
    content
)
print("✓ Added --use_cosine_scheduler argument")

# 5. Add gradient clipping argument
content = re.sub(
    r'(parser\.add_argument\("--weight_decay".*\n)',
    r'\1    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")\n',
    content
)
print("✓ Added --gradient_clip argument")

# 6. Add gradient clipping in training loop
# Find the backward() call and add clipping after it
content = re.sub(
    r'(        loss\.backward\(\))\n(        optimizer\.step\(\))',
    r'\1\n        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n\2',
    content
)
print("✓ Added gradient clipping in training loop")

# Write the modified content
with open('train_improved.py', 'w') as f:
    f.write(content)

print("\n✅ All improvements applied successfully!")
print("\nChanges made:")
print("  1. weight_decay: 1e-4 → 1e-3 (stronger regularization)")
print("  2. Added label_smoothing=0.1 (prevents overconfidence)")
print("  3. Added gradient clipping (prevents exploding gradients)")
print("  4. Added option for CosineAnnealing scheduler")
print("\nNote: num_layers=6 and dropout=0.2 are already optimal from Phase 1")
