use std::collections::VecDeque;

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Bool, Device, Int, Shape, Tensor},
};
use rand::{rngs::ThreadRng, Rng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tracing::{debug, info, instrument, trace, Span};

pub struct MmapRawDataset {
    /// Memory map.
    mmap: memmap2::Mmap,
    /// The number of unsigned 32-bit integers in a block.
    block_size: usize,
}

impl Dataset<Vec<u32>> for MmapRawDataset {
    #[instrument(skip(self))]
    fn get(&self, index: usize) -> Option<Vec<u32>> {
        debug!("Getting data block at index {}", index);
        let st = index << 2;
        let en = st + self.block_size * 4;
        let mucked: &[u32] = bytemuck::cast_slice(&self.mmap[st..en]);
        let result = mucked.to_vec();
        debug!("Retrieved block of size {}", result.len());
        Some(result)
    }

    fn len(&self) -> usize {
        (self.mmap.len() >> 2) - self.block_size
    }
}

#[derive(Debug, Clone)]
pub struct MmapChunkBatcher<B: Backend> {
    device: Device<B>,
    padding_token: u32,
}

#[derive(Debug, Clone)]
pub struct MmapChunkBatch<B: Backend> {
    /// The tensor of size (B, T), i.e, matrix of T-vector token IDs for every training sample.
    pub cat_tensor: Tensor<B, 2, Int>,
    /// The tensor of size (B, T), i.e, matrix of masking vectors for each training sample.
    pub pad_mask: Tensor<B, 2, Bool>,
    /// The target tokens expected (against which loss is measured) in training.
    pub targets: Option<Tensor<B, 1, Int>>,
}

impl<B: Backend> MmapChunkBatch<B> {
    fn from_iter(
        len: usize,
        batch_iter: impl Iterator<Item = MmapChunkBatchElement<B>>,
    ) -> MmapChunkBatch<B> {
        let mut tensors = Vec::with_capacity(len);
        let mut masks = Vec::with_capacity(len);
        let mut targets = Vec::with_capacity(len);

        for elm in batch_iter {
            tensors.push(elm.token_ids);
            masks.push(elm.pad_mask);
            targets.push(elm.targets);
        }

        debug!(
            tl = tensors.len(),
            ml = masks.len(),
            "Will catenate tensors;"
        );

        MmapChunkBatch {
            cat_tensor: Tensor::cat(tensors, 0),
            pad_mask: Tensor::cat(masks, 0),
            targets: Some(Tensor::cat(targets, 0)),
        }
    }
}

struct MmapChunkBatchElement<B: Backend> {
    token_ids: Tensor<B, 2, Int>,
    pad_mask: Tensor<B, 2, Bool>,
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> MmapChunkBatcher<B> {
    /// Take a block of token IDs, apply auto-regressive padding, i.e, for a block of length `N` generate `N` training samples, each of length 1, 2, .., N.
    /// Pre-condition: All inputs must be of size `block_size`. Usually, block_size = max_seq_len + 1
    #[instrument(skip(self, block))]
    fn gen_auto_regressive_pad(&self, block: Vec<u32>) -> MmapChunkBatchElement<B> {
        assert!(!block.is_empty(), "Must provide a non-empty block to pad.");

        let max_length = block.len() - 1;

        let temp: Tensor<B, 1, Int> = Tensor::from_ints(&block[..max_length], &self.device); // last one is always a target or pad.
        let element = temp
            // .one_hot(self.vocab_size), don't one-hot encode
            .unsqueeze::<2>();

        let mut tensors = Tensor::repeat_dim(element, 0, max_length);
        let targets = Tensor::from_ints(&block[1..=max_length], &self.device);

        // use upper triangular mask, so that model always continues from (T-1) seq. position for all inputs.
        let mask = Tensor::triu_mask(Shape::new([max_length, max_length]), 0, &self.device);
        tensors = tensors.mask_fill(mask.clone(), self.padding_token); // is this really needed? (ig Wq, Wk, Wv need to see padding emb.)

        MmapChunkBatchElement {
            token_ids: tensors,
            pad_mask: mask,
            targets,
        }
    }
}

impl<B: Backend> Batcher<Vec<u32>, MmapChunkBatch<B>> for MmapChunkBatcher<B> {
    #[instrument(skip(self, items))]
    fn batch(&self, items: Vec<Vec<u32>>) -> MmapChunkBatch<B> {
        let batch_span = Span::current();
        debug!("Processing batch of size {}", items.len());

        // Output-dim: (#batches, block_size, vocab_size)
        let batch_vec: Vec<_> = items
            .into_par_iter()
            .map(|block| {
                let _guard = batch_span.enter();
                trace!("Converting block to tensors with padding");
                self.gen_auto_regressive_pad(block)
            })
            .collect();

        trace!(
            "Concatenating {} batch elements (tensors and padding masks). This happens serially.",
            batch_vec.len()
        );
        MmapChunkBatch::from_iter(batch_vec.len(), batch_vec.into_iter()) // Serial iteration over the batch elements.
    }
}

/// Creates a new `MmapRawDataset` and `MmapChunkBatcher` for training on a given data file
///
/// # Arguments
///
/// * `fname` - Path to the training data file
/// * `block_size` - Number of tokens in each training block
/// * `vocab_size` - Size of the vocabulary (number of unique tokens)
/// * `device` - The device to run computations on
///
/// # Returns
///
/// A tuple containing:
/// * `MmapRawDataset` - Memory mapped dataset for efficient data loading
/// * `MmapChunkBatcher` - Batcher to convert raw blocks into one-hot tensors
///
/// # Errors
///
/// Returns an IO error if the file cannot be opened or memory mapped
#[instrument(skip(device))]
pub fn open_mmap_dataset<B: Backend>(
    fname: &str,
    block_size: usize,
    padding_token: u32,
    device: Device<B>,
) -> std::io::Result<(MmapRawDataset, MmapChunkBatcher<B>)> {
    info!("Opening memory mapped dataset from {}", fname);

    let file = std::fs::File::open(fname)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    debug!("Successfully memory mapped file of size {}", mmap.len());

    let dataset = MmapRawDataset { mmap, block_size };
    let batcher = MmapChunkBatcher {
        device,
        padding_token,
    };

    debug!("Created dataset with block_size={}", block_size);
    Ok((dataset, batcher))
}

pub struct ChunkSampler<B: Backend> {
    /// Batches queued-up and read from disk already.
    queue: VecDeque<MmapChunkBatch<B>>,
    /// Number of batches to read and maintain.
    q_size: usize,
    batch_size: usize,
    /// Seeded random number generator.
    rng: ThreadRng,
    dataset: MmapRawDataset,
    batcher: MmapChunkBatcher<B>,
}

impl<B: Backend> ChunkSampler<B> {
    pub fn new(
        dataset: MmapRawDataset,
        batcher: MmapChunkBatcher<B>,
        queue_size: usize,
        batch_size: usize,
    ) -> Self {
        let rng = rand::rng();

        Self {
            queue: VecDeque::with_capacity(queue_size),
            q_size: queue_size,
            batch_size,
            rng,
            dataset,
            batcher,
        }
    }
}

impl<B: Backend> Iterator for ChunkSampler<B> {
    type Item = MmapChunkBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.queue.is_empty() {
            let dlen = self.dataset.len();
            let nvars = self.batch_size * self.q_size;
            debug!("Chunk buffer is empty; refilling with {nvars} chunks");

            // has to be generated before hand.
            let rvs: Vec<_> = (0..nvars).map(|_| self.rng.random::<f64>()).collect();

            let v: Vec<_> = rvs
                .into_par_iter()
                .map(|rv| {
                    // potential precision loss.
                    let index = ((rv * (dlen as f64)).floor() as usize).min(dlen - 1);
                    let read = self.dataset.get(index).expect("msg");
                    self.batcher.gen_auto_regressive_pad(read)
                })
                .chunks(self.batch_size)
                .map(|chunk| MmapChunkBatch::from_iter(chunk.len(), chunk.into_iter()))
                .collect();
            trace!("Finished generating batches in parallel.");
            self.queue.extend(v);
        }

        self.queue.pop_front()
    }
}

pub fn good_size_for(division_width: f64, pv: f64) -> usize {
    if division_width <= 1.0 {
        return 1;
    }
    assert!(pv > 0.0, "p-value must be non-negative");
    assert!(pv < 1.0, "pv must be < 1.0");
    let f = (division_width.ln() - pv.ln()) / (division_width.ln() - (division_width - 1.0).ln());
    let r = f.ceil() as usize;
    trace!("Good size for width {division_width} and pv {pv} is {r}");
    r
}
