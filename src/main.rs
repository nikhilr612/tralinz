//! Main entry point.

#![warn(clippy::pedantic)]

mod basemha;
mod tokut;

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use clap::Parser;
use tracing::info;
use tracing_subscriber::FmtSubscriber;

/// Command line arguments for the binary.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct CliArgs {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(clap::Subcommand)]
enum CliCommand {
    /// Tokenize training and validation data (provided as text files) and dump token IDs in binary format.
    PreTokenize {
        /// Path to tokenizer.
        #[arg(short, long)]
        tokenizer_path: String,
        /// Path to text file to tokenize.
        text_file: String,
        /// Path to output file to write tokens to. If not specified, output file is the input path with `.bin` extension.
        #[arg(short, long)]
        output_file: Option<String>,
        /// Enforce endianess, and write token IDs in big endian.
        #[arg(long = "endian")]
        safe_big_endian: bool,
        /// The number of lines per chunk of text processed
        #[arg(short, long, default_value_t = 1024)]
        buffer_line_count: usize,
    },
    /// Randomly sample a block of tokens from `.bin` file and print the decoded text.
    /// The main use-case for this is to verify tokenizer decoding, as well as the pre-tokenized data.
    /// Currently, only native endian token IDs are supported.
    SampleBlock {
        // TODO: Add flag for checking big-endian token IDs.
        /// File containing pre-tokenized data.
        tokenized_data_file: String,
        /// Path to tokenizer.
        #[arg(short, long)]
        tokenizer_path: String,
        /// The size of the block of tokens read from file.
        #[arg(short, long, default_value_t = 256)]
        block_size: u64,
    },
}

fn main() {
    FmtSubscriber::builder().init();

    let args = CliArgs::parse();
    match args.command {
        CliCommand::PreTokenize {
            tokenizer_path,
            text_file,
            output_file,
            safe_big_endian,
            buffer_line_count,
        } => {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap_or_else(|e| {
                panic!("Failed to read tokenizer from file {tokenizer_path}, cause: {e}")
            });
            let input_file = File::open(&text_file).unwrap_or_else(|e| {
                panic!("Failed to open input text file {text_file}, cause: {e}")
            });
            let output_file = File::options()
                .create(true)
                .append(true)
                .open(output_file.map_or_else(
                    || Path::new(&text_file).with_extension("bin"),
                    PathBuf::from,
                ))
                .unwrap_or_else(|e| panic!("Failed to open output file, cause: {e}"));
            info!(text_file, "Starting tokenization");
            tokut::tokenize_file(
                &tokenizer,
                input_file,
                output_file,
                safe_big_endian,
                buffer_line_count,
            )
            .unwrap_or_else(|e| {
                panic!("Tokenization of input file {text_file} failed.\n\tCause: {e}");
            });
        }
        CliCommand::SampleBlock {
            tokenized_data_file,
            tokenizer_path,
            block_size,
        } => {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path).unwrap_or_else(|e| {
                panic!("Failed to read tokenizer from file {tokenizer_path}, cause: {e}")
            });
            let text =
                tokut::sample_text_one(Path::new(&tokenized_data_file), &tokenizer, block_size)
                    .unwrap_or_else(|e| panic!("Failed to sample block from file, cause: {e}"));
            println!("Decoded text:\n{text}");
        }
    }
}
