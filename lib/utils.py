from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import torch
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score, f1_score
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

def padding_function_BERT_GCN(elems):

  """
    Args:
      - elems: tuple of batch items
    
    Return:
      - A dictionary containing:
        - words: batched, padded words tensor all at the same size, using function pad_sequence
        - lemma: batched, padded lemmas tensor all at the same size, using function pad_sequence
        - pos_tags: batched, padded pos-tags tensor all at the same size, using function pad_sequence
        - predicate: batched, padded predicates tensor all at the same size, using function pad_sequence
        - dependency_relations: batched, padded dependency relations tensor all at the same size, using function pad_sequence
        - adj: batched, padded adjacent matrix using function pad
        - degree:  batched, padded degree matrix using function pad
        - pretrained: batched, padded words pretrained tensor all at the same size, using function pad_sequence
        - roles: batched, padded roles tensor all at the same size, using function pad_sequence (pad == -100)
  """

  elem = list(zip(*elems))
  words = elem[0]
  lemma = elem[1]
  pos = elem[2]
  predicate = elem[3]
  dep_rel = elem[4]
  pretrained = elem[5]
  adj = elem[6]
  degree = elem[7]
  labels = elem[8]
  
  words = pad_sequence(words,batch_first=True,padding_value = 0)
  pretrained =pad_sequence(pretrained,batch_first=True,padding_value = 0)
  pos = pad_sequence(pos,batch_first=True,padding_value = 0)
  dep_rel = pad_sequence(dep_rel,batch_first=True,padding_value = 0)
  predicate = pad_sequence(predicate,batch_first=True,padding_value = 0)
  lemma = pad_sequence(lemma,batch_first=True,padding_value = 0)
  labels = pad_sequence(labels,batch_first=True,padding_value = -100)
  new_adj = []
  new_degree =[]

  for i in range(len(adj)):
    N =  lemma.shape[1] - adj[i].shape[1]
    if N > 0:
      A = pad(adj[i],(0,N, 0, N)).unsqueeze(0)
    else:
      A = adj[i].unsqueeze(0)
    new_adj.append(A)

  elem = new_adj[0]
  for i in range(1, len(new_adj)):
    elem = torch.cat((elem, new_adj[i]))

  
  for i in range(len(degree)):
    N =  lemma.shape[1] - degree[i].shape[1]
    if N > 0:
      A = pad(degree[i],(0,N, 0, N)).unsqueeze(0)
    else:
      A = degree[i].unsqueeze(0)
    new_degree.append(A)

  elem_degree = new_degree[0]
  for i in range(1, len(new_degree)):
    elem_degree = torch.cat((elem_degree, new_degree[i]))
  
  item ={
      'words':words,
      'lemma':lemma,
      'pos_tags':pos,
      'predicate':predicate,
      'dependency_relations': dep_rel,
      'adj':elem,
      'degree': elem_degree,
      'pretrained': pretrained,
      'roles': labels
  }
  
  return item


def build_mask(batch_vector: torch.tensor, tokenizer):

    padding_mask = torch.ones_like(batch_vector)
    padding_mask[batch_vector == tokenizer.pad_token_id] = 0
    return padding_mask

def compute_metrics(model:nn.Module, l_dataset:DataLoader, l_label_vocab:dict, tokenizer, device):
    all_predictions = list()
    all_labels = list()
    model.eval()
    for indexed_elem in tqdm(l_dataset):
        indexed_in = indexed_elem["words"].to(device)
        mask = build_mask(indexed_in, tokenizer)
        POS = indexed_elem["pos_tags"].to(device)
        predicate = indexed_elem['predicate'].to(device)
        dep = indexed_elem['dependency_relations'].to(device)
        lemma = indexed_elem['lemma'].to(device)
        adj = indexed_elem['adj'].to(device)
        degree = indexed_elem['degree'].to(device)
        pretrained = indexed_elem['pretrained'].to(device)

        indexed_labels = indexed_elem["roles"].to(device)
        predictions = model(indexed_in,POS,predicate,dep,lemma,pretrained,adj,degree,mask)

        predictions = torch.argmax(predictions, -1).view(-1)
        labels = indexed_labels.view(-1)
        valid_indices = labels != 0

        for target, pred in zip(labels,predictions):
            target = target.item()
            pre = pred.item()
            if target == -100:
              continue
            else:
              all_labels.append(target)
              all_predictions.append(pre)
        
    labels_reverse_index = {v: k for k, v in l_label_vocab.items()}
    all_labels = [labels_reverse_index[target] for target in all_labels]
    all_predictions = [labels_reverse_index[pred] for pred in all_predictions]

    micro_precision = sk_precision(all_labels, all_predictions, average="micro")
    macro_precision = sk_precision(all_labels, all_predictions, average="macro",zero_division=0)
    per_class_precision = sk_precision(all_labels, all_predictions, labels = list(range(len(l_label_vocab))), average=None, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro",zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    return {"micro_precision":micro_precision,
            "macro_precision":macro_precision, 
            "per_class_precision":per_class_precision,
            "f1":f1,
            "recall":recall,
            "all_predictions":all_predictions,
            "all_labels":all_labels}
    

def build_mask_dis(batch_vector: torch.tensor, tokenizer):
        padding_mask = torch.ones_like(batch_vector)
        padding_mask[batch_vector == tokenizer.pad_token_id] = 0
        return padding_mask

def compute_metrics_dis(model:nn.Module, l_dataset:DataLoader, roles_vocab:dict, pred_vocab:dict, tokenizer, device):
    all_predictions_roles = list()
    all_labels_roles = list()
    all_predictions_predi = list()
    all_labels_predi = list()
    for indexed_elem in tqdm(l_dataset):
        indexed_in = indexed_elem["words"].to(device)
        mask = build_mask(indexed_in)
        POS = indexed_elem["pos_tags"].to(device)
        predicate = indexed_elem['predicate'].to(device)
        dep = indexed_elem['dependency_relations'].to(device)
        lemma = indexed_elem['lemma'].to(device)
        adj = indexed_elem['adj'].to(device)
        identify = indexed_elem['identify'].to(device)
        pretrained = indexed_elem['pretrained'].to(device)
        degree = indexed_elem['degree'].to(device)

        roles = indexed_elem["roles"].to(device)
        predictions_pred, predictions_roles = model(indexed_in,POS,dep,pretrained,identify,lemma,adj,degree,mask)

        predictions_pred = torch.argmax(predictions_pred, -1).view(-1)
        predictions_roles = torch.argmax(predictions_roles, -1).view(-1)

        roles = roles.view(-1)
        predicate = predicate.view(-1)

        for target, pred in zip(roles,predictions_roles):
            target = target.item()
            pre = pred.item()
            if target == -100:
              continue
            else:
              all_labels_roles.append(target)
              all_predictions_roles.append(pre)

        for target, pred in zip(predicate,predictions_pred):
            target = target.item()
            pre = pred.item()
            if target == -100:
              continue
            else:
              all_labels_predi.append(target)
              all_predictions_predi.append(pre)
        
    labels_reverse_index_roles = {v: k for k, v in roles_vocab.items()}
    labels_reverse_index_predi = {v: k for k, v in pred_vocab.items()}

    all_labels_roles = [labels_reverse_index_roles[target] for target in all_labels_roles]
    all_predictions_roles = [labels_reverse_index_roles[pred] for pred in all_predictions_roles]

    all_labels_pred = [labels_reverse_index_predi[target] for target in all_labels_predi]
    all_predictions_pred = [labels_reverse_index_predi[pred] for pred in all_predictions_predi]

    # global precision. Does take class imbalance into account.
    micro_precision_roles = sk_precision(all_labels_roles, all_predictions_roles, average="micro")
    micro_precision_pred = sk_precision(all_labels_pred, all_predictions_pred, average="micro")

    # precision per class and arithmetic average of them. Does not take into account class imbalance.
    macro_precision_roles = sk_precision(all_labels_roles, all_predictions_roles, average="macro",zero_division=0)
    macro_precision_pred = sk_precision(all_labels_pred, all_predictions_pred, average="macro",zero_division=0)

    f1_roles = f1_score(all_labels_roles, all_predictions_roles, average="macro",zero_division=0)
    f1_pred = f1_score(all_labels_pred, all_predictions_pred, average="macro",zero_division=0)

    recall_roles = recall_score(all_labels_roles, all_predictions_roles, average="macro", zero_division=0)
    recall_pred = recall_score(all_labels_pred, all_predictions_pred, average="macro", zero_division=0)

    return {"micro_precision_roles":micro_precision_roles,
            "micro_precision_pred":micro_precision_pred,
            "macro_precision_roles":macro_precision_roles, 
            "macro_precision_pred":macro_precision_pred, 
            "f1_roles":f1_roles,
            "f1_pred":f1_pred,
            "recall_roles":recall_roles,
            "recall_pred":recall_pred,
            "all_predictions_roles":all_predictions_roles,
            "all_labels_roles":all_labels_roles,
             "all_predictions_pred":all_predictions_pred,
            "all_labels_pred":all_labels_pred}

