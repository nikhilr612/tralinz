//! Module to contain all command-line argument related structs, and functions.

use core::panic;

use burn::{config::Config, optim::AdamWConfig};
use clap::Parser;
use tracing::trace;

use crate::basemha;

/// Command line arguments for the binary.
#[derive(Parser)]
#[command(version, about, long_about = None)]
pub(crate) struct CliArgs {
    #[command(subcommand)]
    pub(crate) command: CliCommand,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
#[derive(clap::ValueEnum)]
pub(crate) enum ModelVariant {
    /// Plain transformer with mult-head attention.
    BaseMHA,
}

#[derive(clap::Subcommand)]
pub(crate) enum CliCommand {
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
    /// Train a transformer on pre-tokenized data.
    Train {
        /// Model type to train
        variant: ModelVariant,
        /// Output directory for model files.
        artifact_dir: String,
        /// Pre-tokenized training data.
        train_file: String,
        #[arg(short, long)]
        /// Pre-tokenized test data.
        test_file: Option<String>,
        /// Training configuration.
        train_config: String,
    },
    /// Generate default training configuration for specfied model.
    /// Training and Model configurations can be customized by editing the output json file.
    GenerateConfig {
        /// The type of model to train.
        variant: ModelVariant,
        #[arg(short, long, default_value_t = 32768)]
        /// Tokenizer vocabulary size.
        vocab_size: usize,
        /// Output path to write the configuration to.
        output_path: String,
    },
}

impl ModelVariant {
    pub(crate) fn train(
        self,
        train_file: &str,
        test_file: Option<&str>,
        artifact_dir: &str,
        config_path: &str,
    ) {
        match self {
            ModelVariant::BaseMHA => {
                handle_train_base_mha(train_file, test_file, artifact_dir, config_path);
            }
        }
    }

    pub(crate) fn gen_config(self, vocab_size: usize, output_path: &str) {
        match self {
            ModelVariant::BaseMHA => {
                handle_config_base_mha(vocab_size, output_path);
            }
        }
    }
}

fn handle_config_base_mha(vocab_size: usize, output_path: &str) {
    let model_config = basemha::ModelConfig::new(vocab_size);

    let adamwconfig = AdamWConfig::new();
    let training_config = basemha::TrainingConfig::new(model_config, adamwconfig);
    training_config.save(output_path).unwrap_or_else(|e| {
        panic!("Failed to save configuration to file {output_path}.Cause:\n\t{e}");
    });
}

fn handle_train_base_mha(
    train_file: &str,
    test_file: Option<&str>,
    artifact_dir: &str,
    config_path: &str,
) {
    let device = burn_tch::LibTorchDevice::Cuda(0); // burn_cuda::CudaDevice::new(0);
    let training_config = basemha::TrainingConfig::load(config_path).unwrap_or_else(|e| {
        panic!("Failed to load training configuration from {config_path}. Cause:\n\t{e}")
    });

    trace!("Created training configuration.");
    basemha::train::<burn::backend::Autodiff<burn_tch::LibTorch>>(
        artifact_dir,
        train_file,
        test_file,
        training_config,
        device,
    );
}
