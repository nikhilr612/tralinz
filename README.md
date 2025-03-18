# Tralinz
A repository for experimenting with attention mechanisms, transformers, and related neural network architectures using the [burn](https://burn.dev/) deep-learning framework.
This project focuses on implementing small-scale models to explore architectural innovations and gain deeper insights into attention-based approaches.

## Goals
- Build compact transformer models.
- Investigate architectural novelties.
- Document key findings.

## Datasets
Transformers are notoriously data-hungry; notwithstanding datasets used in this project will be limited in size.
1. [OpenWebText 2M Subset](https://www.kaggle.com/datasets/nikhilr612/openwebtext-2m-subset)

## Dependencies
- [burn](https://burn.dev) (0.16.0) - Deep learning framework
- clap (4.5.32) - Command line argument parser
- [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) (0.21.1) - Text tokenization library
- tracing (0.1.41) - Application-level tracing
- tracing-subscriber (0.3.19) - Tracing implementation
- rand (0.9.0) - Random number generation
- bytemuck (1.22.0) - Memory manipulation utilities
