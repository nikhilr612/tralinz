FROM rust:1.86

WORKDIR /usr/lib/

# Get Libtorch 2.6, and extract it.
RUN wget -O libtorch.zip "https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip" \
    && unzip libtorch.zip \
    && rm libtorch.zip
# ^--- reasonable layer size ~4.5GB uncompressed.

# Set up environment variables to use torch.
ENV LIBTORCH=/usr/lib/libtorch
ENV LD_LIBRARY_PATH=/usr/lib/libtorch/lib:$LD_LIBRARY_PATH

# Get around version checks, etc. (burn has yet to support latest torch release)
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /usr/app/tralinz

# Copy the dataset and tokenizer.
COPY ./dataset/uc4full.bin ./dataset.bin
# ^--- largest layer so far.
COPY ./dataset/uc4-unigram-12k.json ./tokenizer.json

# Copy source.
COPY ./src ./src
COPY Cargo.toml .

# Copy custom patch.
COPY ./deps/burn/Cargo.toml ./deps/burn/Cargo.toml
COPY ./deps/burn/crates ./deps/burn/crates

# Install binary
RUN cargo install --path .

CMD ["tralinz"]
