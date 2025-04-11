//! Implement an Encoder-Decoder type transformer with _`MultiHead`_ attention.

use std::path::PathBuf;

use burn::{
    config::Config,
    // data::dataloader::DataLoaderBuilder,
    lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig,
    module::{Module, Param},
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        loss::CrossEntropyLossConfig,
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        LinearConfig,
    },
    optim::AdamWConfig,
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Bool, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use tracing::{debug, debug_span, info, info_span, trace};

use crate::{
    dataset::{open_mmap_dataset, ChunkSampler, MmapChunkBatch},
    learn::ShoddyLearner,
};

#[derive(Debug, Module)]
/// Module for weight tying embedder with head.
struct TiedEmbedder<B: Backend> {
    /// Weight matrix of size `(V, C)` = `(VocabSize, EmbeddingDim)` for embeddings.
    /// The transpose is used as language model head to obtain logits.
    weights: Param<Tensor<B, 2>>,
}

/// Implemented for conformity with the burn way of doing things.
#[derive(Config, Debug)]
struct TiedEmbedderConfig {
    vocab_size: usize,
    embedding_dim: usize,
    initializer: Initializer,
}

impl TiedEmbedderConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> TiedEmbedder<B> {
        let weights = self
            .initializer
            .init([self.vocab_size, self.embedding_dim], device);
        TiedEmbedder { weights }
    }
}

impl<B: Backend> TiedEmbedder<B> {
    /// Map Tensor of Token IDs to their embeddings.
    fn embed(&self, tokenids: Tensor<B, 2, burn::prelude::Int>) -> Tensor<B, 3> {
        burn::tensor::module::embedding(self.weights.val(), tokenids)
    }

    /// Get logits with the penultimate input.
    /// Apply a weights on the input, and return the logits.
    /// The original weights used for embedding are re-used here.
    /// The output has dimensions `(B, T, V)` and must be pooled/sampled for `(B, V)` after which an `argmax` fixes the output token.
    fn prelogits(&self, penult: Tensor<B, 3>) -> Tensor<B, 3> {
        // Penult - (B, T, C) = (Batch, SeqLen, Embedding dim)
        // Output - (B, T, V) = (Batch, SeqLen, Vocab Size)
        let w = self.weights.val().transpose().unsqueeze::<3>();
        penult.matmul(w)
    }
}

#[derive(Debug, Module)]
/// Layer-norm-ed Self-attention (MHA) _without_ masking.
struct ModelBlock<B: Backend> {
    layer_norm: LayerNorm<B>,
    mha: MultiHeadAttention<B>,
}

impl<B: Backend> ModelBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let y = self.layer_norm.forward(x.clone()); // Layer-norm first.
        let mha_output = self.mha.forward(MhaInput::self_attn(y));
        let y = mha_output.context;
        x + y // Skip connection
    }
}

#[derive(Debug, Config)]
struct ModelBlockConfig {
    d_model: usize,
    n_heads: usize,
    #[config(default = "0.1")]
    attention_dropout: f64,
}

impl ModelBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ModelBlock<B> {
        ModelBlock {
            layer_norm: LayerNormConfig::new(self.d_model).init(device),
            mha: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.attention_dropout)
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
struct Model<B: Backend> {
    token_embedder: TiedEmbedder<B>,
    /// "Weights" of shape (T, C)
    position_embeddings: Param<Tensor<B, 2>>,
    masked_mha: MultiHeadAttention<B>,
    blocks: [ModelBlock<B>; 5], // TODO: Add final 2 dense layers if required ("Position-wise FFN").
    finlayer_1: Linear<B>,
    gelu: Gelu,
    finlayer_2: Linear<B>,
    dropout: Dropout,
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Vocabulary size of the tokenizer.
    vocab_size: usize,
    #[config(default = "1536")]
    /// Maximum sequence length
    max_seq_len: usize,
    #[config(default = "768")]
    /// Embedding dimension of the model.
    d_model: usize,
    #[config(default = "8")]
    /// Number of attention heads in multi-head attention.
    n_heads: usize,
    #[config(default = "3072")]
    /// Final Feed-forward network expanding dimension
    d_ffn: usize,
    #[config(default = "0.2")]
    dropout: f64,
    #[config(default = "0.1")]
    attention_dropout: f64,
    /// Initializer for tied embedder and positional encoder.
    #[config(default = "Initializer::Uniform { min: -0.5, max: 0.5 }")]
    // TODO: Use better initializer.
    initializer: Initializer,
    /// Padding token
    #[config(default = "1")]
    padding_token: u32,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let dropout = DropoutConfig::new(self.dropout).init();
        let finlayer_1: Linear<B> = LinearConfig::new(self.d_model, self.d_ffn).init(device);
        let finlayer_2: Linear<B> = LinearConfig::new(self.d_ffn, self.d_model).init(device);
        let token_embedder: TiedEmbedder<B> =
            TiedEmbedderConfig::new(self.vocab_size, self.d_model, self.initializer.clone())
                .init(device);

        let position_embeddings: Param<Tensor<B, 2>> = self.initializer.init(
            [self.max_seq_len, self.d_model],
            // Some(self.max_seq_len),
            // Some(self.d_model),
            device,
        );

        let masked_mha = MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
            .with_dropout(self.attention_dropout)
            .with_quiet_softmax(true)
            .init(device);

        let blocks: [ModelBlock<B>; 5] = std::array::from_fn(|_| {
            ModelBlockConfig::new(self.d_model, self.n_heads) // identical blocks; symmetry broken with initialization.
                .with_attention_dropout(self.attention_dropout)
                .init(device)
        });

        Model {
            token_embedder,
            position_embeddings,
            masked_mha,
            blocks,
            finlayer_1,
            gelu: Gelu::new(),
            finlayer_2,
            dropout,
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, batch: MmapChunkBatch<B>, device: &B::Device) -> Tensor<B, 2> {
        let span = debug_span!("basemha-forward");
        let _guard = span.enter();

        let tokens = batch.cat_tensor;
        let pad_mask = batch.pad_mask;
        // let indices = batch.indices;

        let B = tokens.dims()[0]; // batch size.
        let T = tokens.dims()[1]; // sequence length.

        trace!("Making causal mask.");
        let upper_tri: Tensor<B, 2, Bool> = Tensor::tril_mask([T, T], 0, device);
        let causal_mask: Tensor<B, 3, Bool> = upper_tri.unsqueeze::<3>().repeat_dim(0, B);

        let x = self.token_embedder.embed(tokens); // (B,T) (-> (B,T,V)) -> (B,T,C)

        // position embeddings just work as learned biases in this model.
        // TODO: Explore rotary embeddings.
        let p = self
            .position_embeddings
            .val()
            .unsqueeze::<3>()
            .repeat_dim(0, B);

        let x = self.dropout.forward(x + p); // Apply drop-out
        trace!("Obtained embedding and positional encoding");

        let masked_mha_input = MhaInput::self_attn(x)
            .mask_attn(causal_mask)
            .mask_pad(pad_mask);

        trace!("Forwarding to masked attention..");

        let masked_mha_output = self.masked_mha.forward(masked_mha_input);
        let mut xl = masked_mha_output.context;

        trace!("Forwarding to blocks..");
        for block in &self.blocks {
            xl = block.forward(xl);
        }

        trace!("Applying final FFN with GeLU");
        let xfl = self.finlayer_1.forward(xl);
        let xfl = self.gelu.forward(xfl);
        let xl = self.finlayer_2.forward(xfl);

        // (B, T, V)
        let prelogits = self.token_embedder.prelogits(xl);
        let logits = prelogits.slice([0..B, (T - 1)..T]).squeeze::<2>(1); // grab the next prediction
        trace!("Finished processing. Obtained logits;");

        logits
    }

    pub fn forward_classification(
        &self,
        mut batch: MmapChunkBatch<B>,
        device: &B::Device,
    ) -> ClassificationOutput<B> {
        let span = info_span!("basemha-class-out");
        let _guard = span.enter();

        let targets = batch.targets.take().expect("`forward_classification` should only be used when targets are available. No targets are associated with this batch.");
        let logits = self.forward(batch, device);

        let xloss = CrossEntropyLossConfig::new()
            .init(device) // use cross-entropy with logits directly.
            .forward(logits.clone(), targets.clone()); // use clones since logits, and targets are required.

        ClassificationOutput::new(xloss, logits, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MmapChunkBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MmapChunkBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let span = debug_span!("train_step");
        let _guard = span.enter();

        let device = &item.cat_tensor.device();
        let item = self.forward_classification(item, device);
        trace!("Computed classification results and loss, backwarding...");
        debug!("Mean Training Loss: {}", item.loss.clone().mean());
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MmapChunkBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MmapChunkBatch<B>) -> ClassificationOutput<B> {
        let device = &batch.cat_tensor.device();
        self.forward_classification(batch, device)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,
    #[config(default = "100")]
    pub num_epochs: usize,
    #[config(default = "1")] // use 8
    pub batch_size: usize,
    #[config(default = "4")]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    // TODO: Re-place with custom scheduler with warmup if necessary
    #[config(default = "CosineAnnealingLrSchedulerConfig::new(2.4e-4, 8)")]
    pub lr_scheduler: CosineAnnealingLrSchedulerConfig,
    pub max_iter: Option<usize>,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    train_file: &str,
    test_file: Option<&str>,
    config: TrainingConfig,
    device: B::Device,
) {
    std::fs::remove_dir_all(artifact_dir).ok(); //.expect("Artifact directory should be empty or clearable");
    std::fs::create_dir_all(artifact_dir)
        .expect("Should have appropriate permissions for accessing / creating artifact dir.");
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully.");

    B::seed(config.seed);

    let (train_dataset, train_batcher) = open_mmap_dataset::<B>(
        train_file,
        config.model.max_seq_len + 1,
        config.model.padding_token,
        device.clone(),
    )
    .unwrap_or_else(|e| {
        panic!("Failed to open training dataset. Cause: {e}");
    });

    info!("Loaded train dataset.");

    // let (test_dataset, test_batcher) = open_mmap_dataset::<B>(
    //     test_file,
    //     config.model.max_seq_len + 1,
    //     config.model.padding_token,
    //     device.clone(),
    // )
    // .unwrap_or_else(|e| {
    //     panic!("Failed to open training dataset. Cause: {e}");
    // });

    // info!("Loaded test dataset.");

    // let dataloader_train = DataLoaderBuilder::new(train_batcher)
    //     .batch_size(config.batch_size)
    //     .shuffle(config.seed)
    //     .num_workers(config.num_workers)
    //     .build(train_dataset);

    let train_sampler = ChunkSampler::new(train_dataset, train_batcher, 1, config.batch_size);

    trace!("Created training data loader.");
    // let dataloader_test = DataLoaderBuilder::new(test_batcher)
    //     .batch_size(config.batch_size)
    //     .shuffle(config.seed)
    //     .num_workers(config.num_workers)
    //     .build(test_dataset);

    let lr_schedule = config
        .lr_scheduler
        .init()
        .expect("Cosine learning rate schedule should be valid.");

    // let learner = LearnerBuilder::new(artifact_dir)
    //     .metric_train_numeric(AccuracyMetric::new())
    //     .metric_valid_numeric(AccuracyMetric::new())
    //     .metric_train_numeric(LossMetric::new())
    //     .metric_valid_numeric(LossMetric::new())
    //     .with_file_checkpointer(CompactRecorder::new())
    //     .devices(vec![device.clone()])
    //     .num_epochs(config.num_epochs)
    //     .summary()
    //     .build(
    //         config.model.init::<B>(&device),
    //         config.optimizer.init(),
    //         lr_schedule,
    //     );

    let shoddy_learner = ShoddyLearner::new(
        config.model.init(&device),
        lr_schedule,
        config.optimizer.init(),
        config.num_epochs,
        config.max_iter,
        None,
        PathBuf::from(artifact_dir),
    );

    // let model_trained = learner.fit(dataloader_train, dataloader_test);
    // model_trained
    //     .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    //     .expect("Trained model should be saved successfully");

    shoddy_learner.train(train_sampler, &device);
}
