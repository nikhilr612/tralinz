//! Module to implement custom `Learner`

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use burn::data::dataloader::DataLoader;
use burn::record::CompactRecorder;

use burn::tensor::Device;
use burn::train::checkpoint::{Checkpointer, FileCheckpointer};
use burn::train::TrainStep;
use burn::{
    lr_scheduler::LrScheduler, module::AutodiffModule, optim::Optimizer,
    tensor::backend::AutodiffBackend,
};
use tracing::{debug, info, info_span, trace, warn};

use crate::dataset::{self, ChunkSampler, MmapChunkBatch};

/// A "shoddy" learner performing at most `max_iter` iterations per "epoch".
/// Uses File Checkpointing, and keeps track of the standard metrics.
/// Does not perform validation steps.
pub struct ShoddyLearner<B, S, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    _backend: PhantomData<B>,
    model: M,
    lr_scheduler: S,
    optim: O,
    num_epochs: usize,
    max_iter: Option<usize>,
    checkpoint: Option<usize>,
    artifact_dir: PathBuf,
}

impl<B, S, M, O> ShoddyLearner<B, S, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    // TODO: Add periodic validation steps.
    /// Train the associated model with the provided sampler on the specified device.
    pub fn train<OutputTrain>(mut self, mut train_sampler: ChunkSampler<B>, device: &Device<B>)
    where
        M: TrainStep<MmapChunkBatch<B>, OutputTrain>,
    {
        info!("Fitting model: {}", self.model);

        let recorder = CompactRecorder::new();
        let checkpoint_dir = self.artifact_dir.join("checkpoint");
        let checkpointer_model = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model");
        let checkpointer_optimizer =
            FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim");
        let checkpointer_scheduler = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler");

        trace!("Initialized checkpointers.");

        let start_epoch = match self.checkpoint {
            None => 1,
            Some(chkpt) => {
                info!("Loading checkpoint from {chkpt}");
                let model = checkpointer_model
                    .restore(chkpt, device)
                    .unwrap_or_else(|e| {
                        panic!("Failed to restore model checkpoint from {chkpt},\n\tcause: {e:?}")
                    });
                let optimizer = checkpointer_optimizer
                    .restore(chkpt, device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to restore optimizer checkpoint from {chkpt},\n\tcause: {e:?}"
                        )
                    });
                let scheduler: <S as LrScheduler>::Record<B> = checkpointer_scheduler
                    .restore(chkpt, device)
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to restore scheduler checkpoint from {chkpt},\n\tcause: {e:?}"
                        )
                    });

                self.model = self.model.load_record(model);
                self.optim = self.optim.load_record(optimizer);
                self.lr_scheduler = self.lr_scheduler.load_record(scheduler);
                chkpt + 1
            }
        };

        let max_iter = self
            .max_iter
            .unwrap_or_else(|| dataset::good_size_for(500.0, 0.01)); // cover every 1/500th with 99% chance.

        trace!("Maximum iterations per epoch = {max_iter}");

        for epoch in start_epoch..=self.num_epochs {
            let epoch_span = info_span!("sh-epoch", epoch);
            let _guard = epoch_span.enter();

            trace!("Starting training step.");
            for iteration in 0..max_iter {
                let batch = train_sampler
                    .next()
                    .expect("Sampler should sample indefinitely");
                let lr = self.lr_scheduler.step();
                let output = self.model.step(batch);
                debug!("Forward/Backward-ed batch {iteration} with learning rate {lr}");
                self.model = self.model.optimize(&mut self.optim, lr, output.grads);
                trace!(iteration, "Updated model based on current batch.");
            }

            trace!("Checkpointing..");
            checkpointer_model
                .save(epoch, self.model.clone().into_record())
                .unwrap_or_else(|e| {
                    warn!("Failed to save model checkpoint,\n\tcause:{e:?}");
                });
            debug!("Saved model checkpoint.");
            trace!("Saving optimizer checkpoint..");
            checkpointer_optimizer
                .save(epoch, self.optim.to_record())
                .unwrap_or_else(|e| {
                    warn!("Failed to save optimizer checkpoint,\n\tcause:{e:?}");
                });
            trace!("Saving learning rate scheduler checkpoint..");
            checkpointer_scheduler
                .save(epoch, self.lr_scheduler.to_record::<B>())
                .unwrap_or_else(|e| {
                    warn!("Failed to save scheduler checkpoint,\n\tcause: {e:?}");
                });
        }

        self.model
            .save_file(self.artifact_dir.join("model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    }
}

impl<B, S, M, O> ShoddyLearner<B, S, M, O>
where
    B: AutodiffBackend,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B>,
    S: LrScheduler,
{
    pub fn new(
        model: M,
        lr_scheduler: S,
        optim: O,
        num_epochs: usize,
        max_iter: Option<usize>,
        checkpoint: Option<usize>,
        artifact_dir: PathBuf,
    ) -> Self {
        Self {
            _backend: PhantomData,
            model,
            lr_scheduler,
            optim,
            num_epochs,
            max_iter,
            checkpoint,
            artifact_dir,
        }
    }
}
