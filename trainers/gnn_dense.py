"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch

# Locals
from .gnn_base import GNNBaseTrainer

class DenseGNNTrainer(GNNBaseTrainer):
    """Trainer code for dense GNN."""

    def __init__(self, real_weight=1, fake_weight=1, **kwargs):
        super(DenseGNNTrainer, self).__init__(**kwargs)
        self.real_weight = real_weight
        self.fake_weight = fake_weight

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        # Loop over training batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)

            # Compute target weights on-the-fly for loss function
            batch_weights_real = batch_target * self.real_weight
            batch_weights_fake = (1 - batch_target) * self.fake_weight
            batch_weights = batch_weights_real + batch_weights_fake

            # Train on this batch
            self.model.zero_grad()
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target, weight=batch_weights)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            self.logger.debug('  train batch %i, loss %f', i, batch_loss.item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Current LR %f', summary['lr'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0

        # Loop over batches
        for i, (batch_input, batch_target) in enumerate(data_loader):
            batch_input = [a.to(self.device) for a in batch_input]
            batch_target = batch_target.to(self.device)

            # Make predictions on this batch
            batch_output = self.model(batch_input)
            batch_loss = self.loss_func(batch_output, batch_target).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch_target > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        return summary

def _test():
    t = DenseGNNTrainer(output_dir='./')
    t.build_model()
