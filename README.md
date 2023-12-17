
# Coding Assignment: Implementation and Optimization of GPT-2 Model

### Task1 Model Implementation and Checkpoints

First I started with Exploring Transformers by YouTube Video after that I learn about BERT Architecture referrring this github link mentioned in Refrences Section and implementing class of activation function GELU


```bash
class CustomGELU(nn.Module):

    def forward(self, x):
        return gelu_new(x)
```
Decoder Class having parameters and Hyperparameters first It goes to maked self attention layer after that feed forward Neural Network


```bash
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

```

After implementing Decoder class I implemented GPT class module using DECODER class


```bash

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
```


## Task2 Transformer Architectural Changes

I applied some changes in self attention layer with different approaches to enhance GPT with less and efficient processing power

### Rotary Positional Embedding

This Embedding technique is inbuild in Pytorch can be installed and access using following commands


```bash
$ pip install rotary-embedding-torch
$ from rotary_embedding_torch import RotaryEmbedding
```

And modified in Task1 code in following way :
```bash
self.attention = nn.MultiheadAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True,
            groups=groups,
        )
```

### Group Query Attention
This Embedding technique implemented in following code with reference of following Architecture:

```bash
class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, groups, dropout=0.0, bias=True, batch_first=False):
        super(GroupedQueryAttention, self).__init__()
        self.groups = groups
        self.grouped_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(self, query, key, value, mask=None, need_weights=True):
        group_queries = split_into_groups(query, self.groups)
        group_keys = split_into_groups(key, self.groups)
        group_values = split_into_groups(value, self.groups)
        group_attn_out, _ = self.grouped_attention(group_queries, group_keys, group_values, attn_mask=mask, need_weights=need_weights)
        return group_attn_out
```

And modified in Task1 code in following way :

```bash
self.attention = GroupedQueryAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True,
            groups=groups,
        )
```

### Sliding Window Attention
This Embedding technique implemented in following code with reference of following Architecture:

```bash
class SlidingWindowAttention(nn.Module):
    def __init__(self, window_size, stride, embed_dim, num_heads, dropout=0.0, bias=True, batch_first=False):
        super(SlidingWindowAttention, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )

    def forward(self, query, key, value, mask=None, need_weights=True):
        batch_size, n_tokens, _ = query.shape

        window_indices = torch.arange(0, n_tokens - self.window_size + 1, self.stride)
        outputs = []
        for start_idx in window_indices:
            end_idx = start_idx + self.window_size
            window_query = query[:, start_idx:end_idx, :]
            window_key = key[:, start_idx:end_idx, :]
            window_value = value[:, start_idx:end_idx, :]
            
            
            window_out, _ = self.attention(
                window_query, window_key, window_value,
                attn_mask=mask[:, start_idx:end_idx, start_idx:end_idx] if mask is not None else None,
                need_weights=need_weights
            )
            outputs.append(window_out)

        
        output = torch.cat(outputs, dim=1)

        return output
```

And modified in Task1 code in following way :

```bash
self.attention = SlidingWindowAttention(
            embed_dim=n_embd,
            num_heads=n_head,
            dropout=attn_pdrop,
            bias=True,
            batch_first=True,
            window_size = window_size,
            stride = stride,
        )
```


## Refrences
[minGPT Github Repository](https://github.com/karpathy/minGPT)

[jalammar's Explanation on Transformers Rearch paper Attention all you Need](https://jalammar.github.io/illustrated-transformer/)


[Krish Nayak's Explanation on jalammar's Article](https://www.youtube.com/watch?v=SMZQrJ_L1vo)

[jalammar's Explanation on GPT-2 Rearch paper](
https://jalammar.github.io/illustrated-gpt2/
)

[YouTube Video on Rotarory Embeeding Paper Explaination](
https://www.youtube.com/watch?v=o29P0Kpobz0
)

[YouTube Video on Group Query Embeeding Paper Explaination](
https://www.youtube.com/watch?v=pVP0bu8QA2w
)

[YouTube Video on Sliding Window Attention Paper Explaination](
https://www.youtube.com/watch?v=_8KNb5iqblE
)

