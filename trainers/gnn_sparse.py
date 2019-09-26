"""
This module defines a generic trainer for simple models and datasets.
"""

# Externals
import torch

# Locals
from .gnn_base import GNNBaseTrainer
from utils.ConfusionMatrix import OnlineConfusionMatrix

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0

        # Loop over training batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y, weight=batch.w)
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
        self.lr_scheduler.step()
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
        cm = OnlineConfusionMatrix()

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            # Make predictions on this batch
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            sum_correct += matches.sum().item()
            sum_total += matches.numel()
            cm.update_matrix(batch.y.numpy(), batch_pred.numpy())

            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        summary['f1_score'] = cm.f1_score()
        summary['precision'] = cm.precision()
        summary['recall'] = cm.recall()

        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc']))
        
        self.logger.info('  Validation F1 score : %.3f' %(summary['f1_score']))
        self.logger.info('  Validation Precision score : %.3f' %(summary['precision']))
        self.logger.info('  Validation Recall score : %.3f' %(summary['recall']))
        
        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
