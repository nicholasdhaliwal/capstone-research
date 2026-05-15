# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
#

from dataclasses import dataclass
from typing import Tuple

import torch
from fairseq2.logging import get_log_writer
from torch import Tensor

from lcm.datasets.batch import EmbeddingsBatch, LCMInput, LCMStyle
from lcm.models.abstract_lcm import AbstractLCModel
from lcm.train.criterion import CriterionsFactory
from lcm.train.lcm.criterion import (
    LCMCriterion,
    LCMCriterionConfig,
    compute_standard_mse,
)
from lcm.train.metrics import LossTerm

logger = get_log_writer(__name__)


@dataclass
class ReconstructionCriterionConfig(LCMCriterionConfig):
    min_context_size: int = 1
    """minimum context size for next sentence prediction"""


@CriterionsFactory.register("next_sentence_mse")
class ReconstructionCriterion(LCMCriterion):
    """Computes the MSE reconstruction loss for next-sentence prediction"""

    config: ReconstructionCriterionConfig

    def __init__(
        self,
        config: ReconstructionCriterionConfig,
        model: AbstractLCModel,
        style: LCMStyle = LCMStyle.UNSUPERVISED,
    ):
        super().__init__(config, model, style)

        if style is not LCMStyle.SUPERVISED:
            assert (
                config.min_context_size is not None and config.min_context_size > 0
            ), (
                "For unsupervised pre-training, expecting a min_context_size of at least 1. "
                f"Received min_context_size={config.min_context_size}. "
                "Note that we need some context to predict the first position and "
                "this context can come from a dummy `beginning of document (BOD)` vector. "
                "With a minimum context size of 1 we ensure that we never ask the model to predict BOD"
            )

        self.min_context_size = config.min_context_size

    def prepare_input_and_mask(
        self,
        batch: LCMInput,
    ) -> Tuple[EmbeddingsBatch, torch.Tensor]:
        """
        A method for preparing model inputs and mask for a batch.
        It will be typically reused by the `__call__`
        implementations of the subclasses.
        """
        input_embeddings = batch.prepare_input(style=self.style)

        target_mask = batch.prepare_target_mask(
            input_embeddings,
            style=self.style,
            min_context_size=self.config.min_context_size,
        )

        return input_embeddings, target_mask

    def __call__(self, batch: LCMInput) -> LossTerm:
        """
        Args:
            batch is an LCMInput (see lcm.datasets.batch):

        Returns a LossTerm
        """

        # prepare_input_and mask returns embeddings with seqs in B,T,C
        # and a target mask in B,T,C. Note that the first position is never used as target
        # (i.e. BOS vector or first sentence in the document) and will always be set to False
        # in the target mask
        input_embeddings, target_mask = self.prepare_input_and_mask(batch)

        if self.normalize_in_criterion:
            # the input to the model will be normalize and
            # so is the target used for loss computation
            input_embeddings = input_embeddings.normalize_seqs(self.sonar_normalizer)

        # Predict model outputs
        output_embeddings = self.model(input_embeddings)

        # Prepare predictions and targets:
        # Shift the input to remove the first position.
        # Shifted seqs from input_embeddings are used as ground truth target embeddings
        target_seqs = input_embeddings.seqs[:, 1:].contiguous()
        batch_size, _, sonar_dim = target_seqs.size()

        # shift and flatten
        target_mask = target_mask[:, 1:].reshape(-1)
        # i.e.  s2, s3, s4, s5

        # Trim the last position.
        # output_seqs represent contextualized embeddings / predictions for the next sentence
        # This shifting/trimming allows us to predict `s_t` conditioned on `s_{<t}`
        predicted_seqs = output_embeddings.seqs[:, :-1].contiguous()
        # i.e.  s<=1, s<=2, s<=3, s<=4

        # only measure distance over `target_mask = True` positions
        flattened_predictions = predicted_seqs.view(-1, sonar_dim)[target_mask]
        flattened_target = target_seqs.view(-1, sonar_dim)[target_mask]

        # Cast features to float32 before computing the loss:
        reconstruction_loss, mse_loss = self.compute_loss(
            flattened_predictions.float(), flattened_target.float()
        )

        num_target_elements = target_mask.sum()

        if self.reduction == "sum" or num_target_elements == 0:
            reduced_reconstruction_loss = reconstruction_loss.sum()
            mse_loss = mse_loss.sum()

        elif self.reduction == "mean":
            reduced_reconstruction_loss = reconstruction_loss.mean()
            mse_loss = mse_loss.mean()

        final_loss = reduced_reconstruction_loss

        # Loss summands for records
        summands = {
            "mse_loss": (mse_loss.item(), None),
            "reconstruction_loss": (reduced_reconstruction_loss.item(), None),
        }

        return LossTerm(
            value=final_loss,
            batch_size=batch_size,
            num_target_elements=num_target_elements.item(),
            summands=summands,
        )

    def compute_loss(
        self, flattened_predictions, flattened_target
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes the following loss terms:
            1. The Reconstruction loss we want to optimize as well as:
                2. RMSE loss (for tracking) (in this parent class, RMSE=Reconstruction loss)
            Returns reconstruction_loss, mse_loss
        """
        reconstruction_loss, _ = compute_standard_mse(
            flattened_predictions, flattened_target
        )
        if self.config.compute_rmse:
            epsilon = 1e-5
            reconstruction_loss = torch.sqrt(reconstruction_loss + epsilon)

        return reconstruction_loss, reconstruction_loss


@CriterionsFactory.register("target_mse")
class TargetMSECriterion(ReconstructionCriterion):
    """Computes the LCM training objective given source/target pairs"""

    def __init__(
        self,
        config: ReconstructionCriterionConfig,
        model: AbstractLCModel,
        style: LCMStyle = LCMStyle.SUPERVISED,
    ):
        super().__init__(config, model, style)
