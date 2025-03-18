//! Module to house utilities related to tokenizers.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use rand::Rng;
use tokenizers::Tokenizer;
use tracing::{error, info, info_span, trace};

/// Read `line_count` number of lines and return the buffer with text and bytes read.
fn read_chunk(
    reader: &mut BufReader<File>,
    line_count: usize,
) -> Result<(String, usize), std::io::Error> {
    let mut buffer = String::with_capacity(1024);
    let mut bytes_read = 0;

    for _ in 0..line_count {
        let bytes = reader.read_line(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        bytes_read += bytes;
    }

    Ok((buffer, bytes_read))
}

/// Tokenize input file line-by-line and write generated token IDs to the output file.
pub fn tokenize_file(
    tokenizer: &Tokenizer,
    input_file: File,
    output_file: File,
    safe_big_endian: bool,
    line_count: usize,
) -> Result<(), std::io::Error> {
    let span = info_span!("tokenize_file", safe_big_endian);
    let _guard = span.enter();

    let mut reader = BufReader::new(input_file);
    let mut writer = BufWriter::new(output_file);
    let mut counter = 0;

    loop {
        let (text, bytes_read) = read_chunk(&mut reader, line_count)?;
        if bytes_read == 0 {
            break;
        }

        let span = info_span!("chunk_tokenizer");
        let _guard = span.enter();
        match tokenizer.encode_fast(text, false) {
            Ok(encoding) => {
                let encoding = encoding.get_ids();
                if safe_big_endian {
                    trace!("Converting encoding to big endian.");
                    for token_id in encoding {
                        writer.write_all(&token_id.to_be_bytes())?;
                    }
                } else {
                    trace!("Mucking IDs to bytes.");
                    let mucked = bytemuck::cast_slice(encoding);
                    writer.write_all(mucked)?;
                }
            }
            Err(e) => {
                error!("Failed to tokenize chunk {counter}, cause: {e}");
            }
        }
        info!("Wrote chunk {counter} to output");
        counter += 1;
    }

    writer.flush().unwrap();
    Ok(())
}

pub fn sample_text_one(
    file_path: &Path,
    tokenizer: &Tokenizer,
    block_size: u64,
) -> Result<String, std::io::Error> {
    let span = info_span!("text-sample-one", block_size);
    let _guard = span.enter();

    let metadata = fs::metadata(file_path)?;
    let last_index = metadata.len() - (block_size * 4);

    let mut rng = rand::rng();
    let index = (rng.random_range(0..=last_index) >> 2) << 2;

    let mut bytes = vec![
        0;
        usize::try_from(block_size * 4)
            .expect("4 * block size should be contained in `usize`")
    ];
    let mut file = File::open(file_path)?;

    if file.seek(std::io::SeekFrom::Start(index))? != index {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Seek to random index failed.",
        ));
    }

    trace!("Reading random block");
    file.read_exact(&mut bytes)?;
    let mucked: &[u32] = bytemuck::cast_slice(&bytes);

    info!("Decoding random block of size {block_size} read at index: {index}");
    tokenizer
        .decode(mucked, true)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
}

/// A struct to implement sampling 32-bit aligned chunks from a memory-mapped file.
/// Mem-mapping is considered here for performance reasons only.
struct MmapChunkSampler {
    mmap: memmap2::Mmap,
    /// Size of block to sample.
    block_size: usize,
}

impl MmapChunkSampler {
    /// Create a new sampler from a file path and block size.
    pub fn new(path: &Path, block_size: usize) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Ok(Self { mmap, block_size })
    }

    /// Sample n chunks in parallel, returning a vector of byte vectors.
    pub fn sample_chunks(&self, n: usize) -> Vec<&[u32]> {
        use rayon::prelude::*;

        let max_start = self.mmap.len().saturating_sub(self.block_size * 4);

        (0..n)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                let start = (rng.random_range(0..=max_start) >> 2) << 2;
                let en = start + self.block_size * 4;
                bytemuck::cast_slice(&self.mmap[start..en])
            })
            .collect()
    }
}
