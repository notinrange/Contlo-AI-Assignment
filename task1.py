import torch
import torch.nn as nn

from transformers.activations import gelu_new

''' Using Gaussian Error Linear Unit GELU as activation function''' 
class CustomGELU(nn.Module):

    def forward(self, x):
        return gelu_new(x)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        n_embd, # Dimensionality of the embeddings.
        n_head, # Number of attention heads.
        n_positions, # Maximum number of tokens.
        attn_pdrop, # Probability of dropout on attention weights.
        resid_pdrop, # Probability of dropout after applying the MLP.
        layer_norm_epsilon, # Hyperparameter of layer normalization.
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True,
        )
        '''mask for masked self attention layer'''
        self.register_buffer(
            "mask",
            (1 - torch.tril(torch.ones(n_positions, n_positions))).to(
                dtype=torch.bool
            ),
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            CustomGELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )


    ''' after masked self attention layer forward pass layer is introdued'''
    def forward(self, x):
        batch_size, n_tokens, n_embd = x.shape

        x_ = self.ln_1(x)  # (batch_size, n_tokens, n_embd)

        mask = self.mask[:n_tokens, :n_tokens]  # (n_tokens, n_tokens)

        attn_out, _ = self.attention(
            x_, x_, x_, attn_mask=mask, need_weights=False
        )  # (batch_size, n_tokens, n_embd)
        x = x + attn_out  # (batch_size, n_tokens, n_embd)
        x = x + self.mlp(self.ln_2(x))  # (batch_size, n_tokens, n_embd)

        return x


class GPT(nn.Module):
    
    def __init__(
        self,
        *,
        vocab_size, # Number of tokens in the vocabulary.
        n_layer, # Number of decoder blocks to include.
        n_embd, # Dimensionality of the embeddings.
        n_head, # Number of attention heads.
        n_positions, # Maximum number of tokens.
        attn_pdrop, # Probability of dropout on attention weights.
        embd_pdrop,# Probability of dropout on the sum of embeddings.
        resid_pdrop, # Probability of dropout after applying the MLP.
        layer_norm_epsilon, # Hyperparameter of layer normalization.
    ):
        super().__init__()
        self.n_positions = n_positions
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(n_positions, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        self.blocks = nn.Sequential(
            *[
                Decoder(
                    n_embd=n_embd,
                    n_head=n_head,
                    n_positions=n_positions,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        
        batch_size, n_tokens = idx.shape
        device = idx.device

        if n_tokens > self.n_positions:
            raise ValueError("There are too many tokens in the input")

        positions = torch.arange(n_tokens, device=device)  # (n_tokens,)

        token_emb = self.token_emb(idx)  # (batch_size, n_tokens, n_embd)
        pos_emb = self.pos_emb(positions)[None, ...]  # (1, n_tokens, n_embd)
        x = self.drop(token_emb + pos_emb)  # (batch_size, n_tokens, n_embd)
        x = self.blocks(x)  # (batch_size, n_tokens, n_embd)
        x = self.ln(x)  # (batch_size, n_tokens, n_embd)
        logits = self.head(x)  # (batch_size, n_tokens, vocab_size)

        return logits