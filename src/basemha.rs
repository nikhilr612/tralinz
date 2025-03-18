//! Implement an Encoder-Decoder type transformer with MultiHead attention.

#[derive(Debug, Clone)]
struct ModelConfig {
    /// The number of tokens in the tokenizer vocabulary.
    n_tokens: u64,
    /// The dimension of token embeddings.
    embedding_dim: usize,
    /// The number of attention heads per "block". Must divide `embedding_dim`
    head_count: usize,
}
