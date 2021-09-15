import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import  f1_score

class TrainerBERT_GCN():

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        tokenizer,
        device,
        label_vocab: dict,
        log_steps:int=100,
        log_level:int=2):
        
        """
        Args:
            - model: the model we want to train.
            - loss_function: the loss_function to minimize.
            - optimizer: the optimizer used to minimize the loss_function.
            - tokenizer: BERT tokenizer
            - label_vocab (dictionary): vocabulary for the labels
            - log_steps (int): Number of iterations that we use to observe the loss function trend.
            - log_level (int): Always use to observe the loss function trend
        """
        
        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.tokenizer = tokenizer

        self.label_vocab = label_vocab
        self.log_steps = log_steps
        self.log_level = log_level

    def build_mask(self, batch_vector: torch.tensor):
        padding_mask = torch.ones_like(batch_vector)
        padding_mask[batch_vector == self.tokenizer.pad_token_id] = 0
        return padding_mask
  

    def train(self, train_dataset:Dataset, 
              valid_dataset:Dataset, 
              epochs:int=1):
        
        """
        Args:
            - train_dataset: a Dataset or DatasetLoader instance containing
                             the training instances.
            - valid_dataset: a Dataset or DatasetLoader instance used to evaluate
                             learning progress.
            - epochs: the number of times to iterate over train_dataset.

        Returns:
            - avg_train_loss: the average training loss on train_dataset over epochs.
        """
        
        assert epochs > 1 and isinstance(epochs, int)

        train_loss = 0.0
        best_loss = 3000
        patience = 0
        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()
            train_iterator = tqdm(train_dataset)

            for step, sample in enumerate(train_iterator):
                
                words = sample['words'].to(self.device)
                POS = sample['pos_tags'].to(self.device)
                pred = sample['predicate'].to(self.device)
                dep = sample['dependency_relations'].to(self.device)
                lemma = sample['lemma'].to(self.device)
                labels = sample['roles'].to(self.device)
                pretrained = sample['pretrained'].to(self.device)
                adj = sample['adj'].to(self.device)
                degree = sample['degree'].to(self.device)
                mask = self.build_mask(words)

                self.optimizer.zero_grad()
                predictions = self.model(words,POS,pred,dep,lemma,pretrained,adj,degree,mask)

                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
               
                sample_loss = self.loss_function(predictions, labels)
                sample_loss.backward()
                self.optimizer.step()

                epoch_loss += sample_loss.tolist()
                avg_loss = epoch_loss / (step + 1)
                train_iterator.set_description('Train Avg loss: {:.4f}'.format(avg_loss))


                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss = {:0.4f}'.format(epoch, step, epoch_loss / (step + 1)))
            
            avg_epoch_loss = epoch_loss / len(train_dataset)

            train_loss += avg_epoch_loss
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch, avg_epoch_loss))

            valid_loss, best_loss, patience = self.evaluate(valid_dataset, best_loss, patience)
            
            if self.log_level > 0:
                print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch, valid_loss))

        if self.log_level > 0:
            print('... Done!')
        
        avg_epoch_loss = train_loss / epochs
        return avg_epoch_loss
    

    def evaluate(self, valid_dataset, best_loss, patience):
        
        """
        Args:
            - valid_dataset: the dataset to use to evaluate the model.
            - epoch: epoch

        Returns:
            - avg_valid_loss: the average validation loss over valid_dataset.
        """

        valid_loss = 0.0
        self.model.eval()
        all_labels = list()
        all_predictions = list()
        with torch.no_grad():
            valid_iterator = tqdm(valid_dataset)
            for step,  sample in enumerate(valid_iterator):
                words = sample['words'].to(self.device)
                POS = sample['pos_tags'].to(self.device)
                pred = sample['predicate'].to(self.device)
                dep = sample['dependency_relations'].to(self.device)
                adj = sample['adj'].to(self.device)
                pretrained = sample['pretrained'].to(self.device)
                lemma = sample['lemma'].to(self.device)
                degree = sample['degree'].to(self.device)

                labels = sample['roles'].to(self.device)
                mask = self.build_mask(words)

                predictions = self.model(words,POS,pred,dep,lemma,pretrained,adj,degree,mask)

                ris = torch.argmax(predictions, -1).view(-1)

                predictions = predictions.view(-1, predictions.shape[-1])
                labels = labels.view(-1)
                sample_loss = self.loss_function(predictions, labels)  
                valid_loss += sample_loss.tolist()

                current_avg_loss = valid_loss / (step + 1)

                valid_iterator.set_description('Valid Avg loss: {:.4f}'.format(current_avg_loss))

                for target, pred in zip(labels,ris):
                  target = target.item()
                  pre = pred.item()
                  if target == -100:
                    continue
                  else:
                    all_labels.append(target)
                    all_predictions.append(pre)
                                    
            f1 = f1_score(all_labels, all_predictions, average="macro",zero_division=0)
            print(f"\033[1mf1 score: {f1} \033[0m")        
            
            if (valid_loss / len(valid_dataset)) < best_loss:
                best_loss = valid_loss / len(valid_dataset)
                patience = 0
                torch.save(self.model.state_dict(), 'models/best_model.pt')
                print(f"\033[Improvement perfomances, model saved\033[0m")
            else:
                patience += 1 
  
        return valid_loss / len(valid_dataset), best_loss, patience