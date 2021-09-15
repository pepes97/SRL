import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import  f1_score

class TrainerBERT_GCN_DIS():

    def __init__(
        self,
        model: nn.Module,
        loss_function,
        optimizer,
        tokenizer,
        device,
        role_vocab: dict,
        predi_vocab:dict,
        log_steps:int=100,
        log_level:int=2):
        
        """
        Args:
            - model: the model we want to train.
            - loss_function: the loss_function to minimize.
            - optimizer: the optimizer used to minimize the loss_function.
            - tokenizer: BERT tokenizer
            - role_vocab (dictionary): vocabulary for roles
            - predi_vocab (dictionary): vocabulary for predicates
            - log_steps (int): Number of iterations that we use to observe the loss function trend.
            - log_level (int): Always use to observe the loss function trend
        """
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device =device

        self.role_vocab = role_vocab
        self.pred_vocab = predi_vocab
        self.log_steps = log_steps
        self.log_level = log_level
        self.tokenizer = tokenizer
    
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
            - avg_epoch_loss_roles: the average training loss of roles on train_dataset over epochs.
            - avg_epoch_loss_predi: the average training loss of predicates on train_dataset over epochs.
        """
        
        assert epochs > 1 and isinstance(epochs, int)
        if self.log_level > 0:
            print('Training ...')
        train_loss_roles= 0.0
        train_loss_predi= 0.0
        best_loss = 3000
        for epoch in range(epochs):
            if self.log_level > 0:
                print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss_roles = 0.0
            epoch_loss_predi = 0.0
            self.model.train()
            train_iterator = tqdm(train_dataset)
            for step, sample in enumerate(train_iterator):
                
                words = sample['words'].to(self.device)
                roles = sample['roles'].to(self.device)
                predicate = sample['predicate'].to(self.device)
                POS = sample['pos_tags'].to(self.device)
                dep = sample['dependency_relations'].to(self.device)
                lemma = sample['lemma'].to(self.device)
                identify = sample['identify'].to(self.device)
                degree = sample['degree'].to(self.device)
                adj = sample['adj'].to(self.device)
                pretrained = sample['pretrained'].to(self.device)

                mask = self.build_mask(words)

                self.optimizer.zero_grad()
                predictions_pre,predictions_roles = self.model(words,POS,dep,pretrained,identify, lemma,adj,degree,mask)

                predictions_pre = predictions_pre.view(-1, predictions_pre.shape[-1])
                predictions_roles = predictions_roles.view(-1, predictions_roles.shape[-1])
           
                roles = roles.view(-1)
                predicate = predicate.view(-1)
               
                sample_loss_roles = self.loss_function(predictions_roles, roles)
                sample_loss_predicate = self.loss_function(predictions_pre, predicate)
                loss = sample_loss_roles + sample_loss_predicate
                loss.backward()
                self.optimizer.step()

                epoch_loss_roles += sample_loss_roles.tolist()
                avg_loss_roles = epoch_loss_roles / (step + 1)
                epoch_loss_predi += sample_loss_predicate.tolist()
                avg_loss_predi = epoch_loss_predi / (step + 1)

                train_iterator.set_description('Train Avg loss Roles: {:.4f}'.format(avg_loss_roles))


                if self.log_level > 1 and step % self.log_steps == self.log_steps - 1:
                    print('\t[E: {:2d} @ step {}] current avg loss roles = {:0.4f}, current avg loss pred = {:0.4f}'.format(epoch, step, epoch_loss_roles / (step + 1), epoch_loss_predi / (step + 1)))

                    
            avg_epoch_loss_roles = epoch_loss_roles / len(train_dataset)
            avg_epoch_loss_predi = epoch_loss_predi / len(train_dataset)
            train_loss_roles += avg_epoch_loss_roles
            train_loss_predi += avg_epoch_loss_predi
            
            if self.log_level > 0:
                print('\t[E: {:2d}] train loss roles = {:0.4f}, train loss predi = {:0.4f}'.format(epoch, avg_epoch_loss_roles, avg_epoch_loss_predi))

            valid_loss_roles, valid_loss_predi, best_loss, patience = self.evaluate(valid_dataset,best_loss, patience)
            
            if self.log_level > 0:
                print('  [E: {:2d}] valid loss roles = {:0.4f}, valid loss predi = {:0.4f}'.format(epoch, valid_loss_roles, valid_loss_predi))

        if self.log_level > 0:
            print('... Done!')
        
        avg_epoch_loss_roles = train_loss_roles / epochs
        avg_epoch_loss_predi = train_loss_predi / epochs
        return avg_epoch_loss_roles, avg_epoch_loss_predi
    

    def evaluate(self, valid_dataset,best_loss, patience):
        
        """
        Args:
            - valid_dataset: the dataset to use to evaluate the model.
            - epoch: epochs

        Returns:
            - avg_valid_loss_roles: the average validation loss of roles over valid_dataset.
            - avg_valid_loss_predi: the average validation loss of predicates over valid_dataset.
        """

        valid_loss_roles = 0.0
        valid_loss_predi = 0.0
        all_labels_roles = list()
        all_predictions_roles = list()
        all_labels_predi = list()
        all_predictions_predi = list()
        self.model.eval()
        with torch.no_grad():
            valid_iterator = tqdm(valid_dataset)
            for step,  sample in enumerate(valid_iterator):
                words = sample['words'].to(self.device)
                POS = sample['pos_tags'].to(self.device)
                dep = sample['dependency_relations'].to(self.device)
                lemma = sample['lemma'].to(self.device)
                roles = sample['roles'].to(self.device)
                predicate = sample['predicate'].to(self.device)
                mask = self.build_mask(words)
                identify = sample['identify'].to(self.device)
                adj = sample['adj'].to(self.device)
                pretrained = sample['pretrained'].to(self.device)
                degree = sample['degree'].to(self.device)


                predictions_pre, predictions_roles = self.model(words,POS,dep,pretrained, identify,lemma,adj,degree,mask)

                ris_roles = torch.argmax(predictions_roles, -1).view(-1)
                ris_predi = torch.argmax(predictions_pre, -1).view(-1)
                predictions_pre =  predictions_pre.view(-1, predictions_pre.shape[-1])
                predictions_roles = predictions_roles.view(-1, predictions_roles.shape[-1])
                roles = roles.view(-1)
                predicate = predicate.view(-1)
                sample_loss_roles = self.loss_function(predictions_roles, roles)
                sample_loss_predi = self.loss_function(predictions_pre,predicate)  
                valid_loss_roles += sample_loss_roles.tolist()
                valid_loss_predi += sample_loss_predi.tolist()

                current_avg_loss_roles = valid_loss_roles / (step + 1)
                current_avg_loss_predi = valid_loss_predi / (step + 1)

                for target, pred in zip(roles,ris_roles):
                  target = target.item()
                  pre = pred.item()
                  if target == -100:
                    continue
                  else:
                    all_labels_roles.append(target)
                    all_predictions_roles.append(pre)
                
                for target, pred in zip(predicate,ris_predi):
                  target = target.item()
                  pre = pred.item()
                  if target == -100:
                    continue
                  else:
                    all_labels_predi.append(target)
                    all_predictions_predi.append(pre)
                
           
            f1_roles = f1_score(all_labels_roles, all_predictions_roles, average="macro",zero_division=0)
            print(f"\033[1mf1 score roles: {f1_roles} \033[0m")

            f1_predi = f1_score(all_labels_predi, all_predictions_predi, average="macro",zero_division=0)
            print(f"\033[1mf1 score predicates: {f1_predi} \033[0m")

            if (valid_loss_predi / len(valid_dataset)) < best_loss:
                best_loss = valid_loss_predi / len(valid_dataset)
                patience = 0
                torch.save(self.model.state_dict(), 'models/best_model_dis.pt')
                print(f"\033[Improvement perfomances, model saved\033[0m")
            else:
                patience += 1 
           

        return valid_loss_roles / len(valid_dataset), valid_loss_predi / len(valid_dataset), best_loss, patience