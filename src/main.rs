//! Main entry point.

#![warn(clippy::pedantic)]

mod args;
mod basemha;
mod dataset;
mod learn;
mod tokut;

use std::{
    fs::File,
    path::{Path, PathBuf},
};

use args::{CliArgs, CliCommand};
use clap::Parser;
use tracing::info;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

fn main() {
    FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = CliArgs::parse();
    match args.command {
        CliCommand::PreTokenize {
            tokenizer_path,
            text_file,
            output_file,
            safe_big_endian,
            buffer_line_count,
        } => handle_pre_tokenize(
            &tokenizer_path,
            &text_file,
            output_file,
            safe_big_endian,
            buffer_line_count,
        ),
        CliCommand::SampleBlock {
            tokenized_data_file,
            tokenizer_path,
            block_size,
        } => handle_sample_block(&tokenized_data_file, &tokenizer_path, block_size),
        CliCommand::Train {
            variant,
            train_file,
            test_file,
            artifact_dir,
            train_config,
        } => {
            variant.train(
                &train_file,
                test_file.as_deref(),
                &artifact_dir,
                &train_config,
            );
        }
        CliCommand::GenerateConfig {
            variant,
            vocab_size,
            output_path,
        } => {
            variant.gen_config(vocab_size, &output_path);
        }
    }
}

fn handle_pre_tokenize(
    tokenizer_path: &str,
    text_file: &str,
    output_file: Option<String>,
    safe_big_endian: bool,
    buffer_line_count: usize,
) {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap_or_else(|e| {
        panic!("Failed to read tokenizer from file {tokenizer_path}, cause: {e}")
    });
    let input_file = File::open(text_file)
        .unwrap_or_else(|e| panic!("Failed to open input text file {text_file}, cause: {e}"));
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

fn handle_sample_block(tokenized_data_file: &str, tokenizer_path: &str, block_size: u64) {
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap_or_else(|e| {
        panic!("Failed to read tokenizer from file {tokenizer_path}, cause: {e}")
    });
    let text = tokut::sample_text_one(Path::new(tokenized_data_file), &tokenizer, block_size)
        .unwrap_or_else(|e| panic!("Failed to sample block from file, cause: {e}"));
    println!("Decoded text:\n{text}");
}
