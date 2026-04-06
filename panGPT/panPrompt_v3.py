import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import argparse
from torch import nn
import math

def print_banner():
    banner = """
    **************************************************
    *                                                *
    *        Transformer Model Token Prediction      *
    *        panPrompt v0.03 (with repetition penalty)*
    *        author: James McInerney (modified)      *
    *                                                *
    **************************************************
    """
    print(banner)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :, :].transpose(0, 1)
        return self.dropout(x)

class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_length, dropout_rate=0.1, pe_max_len=5000, pe_dropout_rate=0.1):
        super(SimpleTransformerModel, self).__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, pe_max_len, dropout=pe_dropout_rate)
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return self.out(x)

def predict_next_tokens(model, tokenizer, prompt, num_tokens, temperature=1.0, top_k=50, top_p=0.95, repetition_penalty=1.0):
    """
    Improved token prediction with top-k, top-p, and repetition penalty
    
    Args:
        top_k: Keep only top k tokens with highest probability (0 = disabled)
        top_p: Keep top tokens with cumulative probability >= top_p (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize repetition)
    """
    model.eval()
    device = next(model.parameters()).device
    tokens = tokenizer.encode(prompt).ids
    
    for _ in range(num_tokens):
        input_ids = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            outputs = model(input_ids)
        
        # Get logits for the last token
        logits = outputs[0, -1, :]
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(tokens):
                # If the token has already appeared, reduce its probability
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty
        
        # Apply temperature
        scaled_logits = logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            scaled_logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probabilities = F.softmax(scaled_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, 1).item()
        tokens.append(next_token_id)
    
    return tokenizer.decode(tokens)

def read_prompt_file(file_path):
    with open(file_path, 'r') as file:
        prompt = file.read().strip()
    return prompt

def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Token prediction with top-k, top-p, and repetition penalty.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--model_type", type=str, required=True, choices=['transformer', 'reformer'], help="Type of model.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the text file containing the prompt.")
    parser.add_argument("--num_tokens", type=int, required=True, help="Number of tokens to predict.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for prediction.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0 to disable).")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling (1.0 to disable).")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (1.0 = no penalty, 1.2 = mild, 1.5 = strong).")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    
    # Initialize model
    if args.model_type == 'transformer':
        model = SimpleTransformerModel(
            vocab_size, 
            args.embed_dim, 
            args.num_heads, 
            args.num_layers, 
            args.max_seq_length,
            dropout_rate=args.dropout_rate,
            pe_max_len=5000,
            pe_dropout_rate=args.dropout_rate
        )
    else:
        raise ValueError("Only transformer model is supported in this version")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    
    # Read prompt
    prompt = read_prompt_file(args.prompt_file)
    print(f"Prompt: {prompt}")
    
    # Generate predictions
    predicted_text = predict_next_tokens(
        model, tokenizer, prompt, 
        args.num_tokens, 
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty
    )
    
    print(f"Predicted text: {predicted_text}")

if __name__ == "__main__":
    main()
