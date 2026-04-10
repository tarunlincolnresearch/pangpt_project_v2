#!/usr/bin/env python3
"""
Check what 50% accuracy actually means with 52K vocabulary
"""

vocab_size = 51999
test_accuracy = 0.5027

print("=== UNDERSTANDING THE ACCURACY ===\n")
print(f"Vocabulary size: {vocab_size:,}")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Random guessing accuracy (1/{vocab_size}): {(1/vocab_size)*100:.4f}%")
print()

print("🤔 WAIT - Something is VERY wrong!")
print(f"   Random guessing should give: ~0.002% accuracy")
print(f"   But we're getting: 50.27% accuracy")
print()

print("💡 HYPOTHESIS:")
print("   The 'accuracy' metric might be calculated incorrectly!")
print("   OR the model is predicting from a MUCH smaller set")
print()

# Check if it's token-level vs sequence-level
print("Let's check if it's token-level accuracy:")
print(f"   Window size: 256 tokens")
print(f"   If model predicts 128 tokens correctly out of 256:")
print(f"   Token accuracy = 128/256 = 50%")
print()
print("   This would mean the model is getting HALF the tokens right")
print("   in each sequence, which is actually GOOD!")
print()

print("🔍 NEED TO CHECK:")
print("   1. Is accuracy calculated per-token or per-sequence?")
print("   2. What is the model actually predicting?")
print("   3. Check the validation code in train_moderate.py")

