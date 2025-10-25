"""Chapter 2"""
import torch
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
, dtype=torch.float32)

#attention with weights  key query and value 

#lets calulate for one one input lets take  "Journey"
x_2= inputs[1]

#get the dimension of input 
d_in=x_2.shape[0]
d_out=2
#create a random weight matrix for key, query, and value
w_key=torch.nn.Parameter(torch.randn(d_in, d_out)) #torch parameter is a wrapper and it is trainable    
w_query=torch.nn.Parameter(torch.randn(d_in, d_out)) #torch parameter is a wrapper and it is trainable
w_value=torch.nn.Parameter(torch.randn(d_in, d_out)) #torch parameter is a wrapper and it is trainable

#calculate the key, query, and value
key=inputs @ w_key
query=inputs @ w_query
value=inputs @ w_value

print(key)
print(query)
print(value)

#let calculate attention scores for one query 

key_2=key[1]
query_2=query[1]

attention_scores_22= torch.dot(key_2, query_2) 
#what does the above the line means ?
# i means the the query of of second attends to itself 

attention_scores_2= query_2 @ key.T

atention_weights_2= torch.softmax(attention_scores_2/torch.sqrt(torch.tensor(d_out)), dim=0) #normalization by the square root of the dimension of the key and query

#lets calculate context vector for the second input
context_vector_2= value @ atention_weights_2

import torch.nn as nn

class SelfAttention_v1(nn.Module):
      def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

      def forward(self, x):
            keys=x@self.W_key
            queries=x@self.W_query
            values=x@self.W_value
            attn_scores=queries@keys.T
            attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
            context_vec=attn_weights@values
            return context_vec

class SelfAttention_v2(nn.Module):
      def __init__(self, d_in, d_out,qkw_bias=False):
            super().__init__()
            self.w_key=nn.Linear(d_in, d_out, bias=qkw_bias)
            self.w_query=nn.Linear(d_in, d_out, bias=qkw_bias)
            self.w_value=nn.Linear(d_in, d_out, bias=qkw_bias)

      def forward(self, x):
            keys=self.w_key(x)
            queries=self.w_query(x)
            values=self.w_value(x)
            attn_scores=queries@keys.T
            attn_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
            context_vec=attn_weights@values
            return context_vec

#lets implement causal self attention masked self attention

sel_v2=SelfAttention_v2(d_in=3, d_out=2, qkw_bias=False)
context_vectors=sel_v2(inputs)
print(context_vectors)

masked=torch.tril(torch.ones(inputs.shape[0], inputs.shape[0]))
print(masked)
#[1. 0. 0. 0. 0. 0.
# 1. 1. 0. 0. 0. 0.
# 1. 1. 1. 0. 0. 0.
# 1. 1. 1. 1. 0. 0.
# 1. 1. 1. 1. 1. 0.
# 1. 1. 1. 1. 1. 1.]

context_vectors_masked=context_vectors*masked
print(context_vectors_masked)

#now lets apply dropout to the attention weights

attention_weights=torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
attention_weights=torch.dropout(attention_weights, p=0.5)
print(attention_weights)

context_vec=attention_weights@values
return context_vec

#implementing compact causal self attention

class CausalAttention(nn.Module):
    
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        
        self.d_out = d_out
        
        # Query, Key, Value projections
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask (prevents looking at future tokens)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_tokens, d_in)
        b, num_tokens, d_in = x.shape
        
        # Project to Q, K, V
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # Compute attention scores: Q @ K^T
        attn_scores = queries @ keys.transpose(1, 2)
        
        # Apply causal mask (replace future positions with -inf)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], 
            -torch.inf
        )
        
        # Scaled softmax (√d_out scaling for stability)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, 
            dim=-1
        )
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        context_vec = attn_weights @ values
        
        return context_vec









