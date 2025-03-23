use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::Backend,
    tensor::{Bool, Device, Int, Shape, Tensor},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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
    /// The indices of final sequence embedding to sample.
    pub indices: Tensor<B, 1, Int>,
}

impl<B: Backend> FromIterator<MmapChunkBatchElement<B>> for MmapChunkBatch<B> {
    #[instrument(skip(iter))]
    fn from_iter<T: IntoIterator<Item = MmapChunkBatchElement<B>>>(iter: T) -> Self {
        let mut tensors = Vec::new();
        let mut masks = Vec::new();
        let mut targets = Vec::new();
        let mut indices = Vec::new();

        for elm in iter {
            tensors.push(elm.token_ids);
            masks.push(elm.pad_mask);
            targets.push(elm.targets);
            indices.push(elm.indices);
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
            indices: Tensor::cat(indices, 0),
        }
    }
}

struct MmapChunkBatchElement<B: Backend> {
    token_ids: Tensor<B, 2, Int>,
    pad_mask: Tensor<B, 2, Bool>,
    targets: Tensor<B, 1, Int>,
    indices: Tensor<B, 1, Int>,
}

impl<B: Backend> MmapChunkBatcher<B> {
    /// Take a block of token IDs, apply auto-regressive padding, i.e, for a block of length `N` generate `N` training samples, each of length 1, 2, .., N.
    /// Pre-condition: All inputs must be of size `block_size`. Usually, block_size = max_seq_len + 1
    #[instrument(skip(self, block))]
    fn gen_auto_regressive_pad(&self, mut block: Vec<u32>) -> MmapChunkBatchElement<B> {
        let mut tensors = Vec::new();
        let mut targets = Vec::new();

        let max_length = block.len() - 1;

        for i in max_length..0 {
            targets.push(block[i]);
            block[i] = self.padding_token;

            let temp: Tensor<B, 1, Int> = Tensor::from_ints(&block[..max_length], &self.device); // last one is always a target or pad.
            let element = temp
                // .one_hot(self.vocab_size), don't one-hot encode
                .unsqueeze::<2>();
            // .float();
            tensors.push(element);
        }

        tensors.reverse();
        let mask = Tensor::tril_mask(Shape::new([max_length, max_length]), 0, &self.device);
        debug!(tl = tensors.len(), "Will catenate tensors;");
        let tensors = Tensor::cat(tensors, 0);
        let targets: Tensor<B, 1, Int> = Tensor::from_ints(&targets[..], &self.device);
        let indices: Tensor<B, 1, Int> = Tensor::arange(
            0..(max_length.try_into().expect("max length should fit in i64")),
            &self.device,
        );

        MmapChunkBatchElement {
            token_ids: tensors,
            pad_mask: mask,
            targets,
            indices,
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
                trace!("Converting block to one-hot tensors with padding");
                self.gen_auto_regressive_pad(block)
            })
            .collect();

        trace!(
            "Stacking {} batch elements (tensors and padding masks). This happens serially.",
            batch_vec.len()
        );
        MmapChunkBatch::from_iter(batch_vec) // Serial iteration over the batch elements.
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
