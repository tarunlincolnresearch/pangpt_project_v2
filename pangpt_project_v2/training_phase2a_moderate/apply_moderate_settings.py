#!/usr/bin/env python3
import re

with open('train_moderate.py', 'r') as f:
    content = f.read()

print("Applying moderate regularization settings...")

# 1. Reduce label smoothing from 0.1 to 0.05
content = re.sub(
    r'CrossEntropyLoss\(label_smoothing=0\.1\)',
    'CrossEntropyLoss(label_smoothing=0.05)',
    content
)
print("✓ Label smoothing: 0.1 → 0.05")

# 2. Reduce weight decay default from 1e-3 to 5e-4
content = re.sub(
    r'--weight_decay.*default=1e-3',
    '--weight_decay", type=float, default=5e-4',
    content
)
print("✓ Weight decay: 1e-3 → 5e-4")

# 3. Reduce dropout default from 0.2 to 0.15
content = re.sub(
    r'--model_dropout_rate.*default=0\.2',
    '--model_dropout_rate", type=float, default=0.15',
    content
)
print("✓ Dropout: 0.2 → 0.15")

# 4. Update version string
content = re.sub(
    r'VERSION = ".*Phase 2 Improved"',
    'VERSION = "0.10a - Phase 2A Moderate"',
    content
)
print("✓ Updated version string")

with open('train_moderate.py', 'w') as f:
    f.write(content)

print("\n✅ Phase 2A settings applied!")
print("\nModerate Regularization:")
print("  - Label smoothing: 0.05 (was 0.1)")
print("  - Weight decay: 0.0005 (was 0.001)")
print("  - Dropout: 0.15 (was 0.2)")
print("  - Gradient clipping: 1.0 (unchanged)")
