import torch
import json
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
from scipy.linalg import fractional_matrix_power

class SvevaDatasetBERT_GCN_DIS(Dataset):
  def __init__(self,
               path,
               tokenizer,
               w_vocab = None,
               l_vocab=None,
               pos_vocab=None,
               pred_vocab=None,
               dep_rel_vocab = None,
               lemma_vocab = None):
    

    """
      Args:
        - path (string): the path to the dataset to be loaded
        - tokenizer: tokenizer of BERT
        - w_vocab (dictionary): vocabulary of words
        - l_vocab (dictionary): vocabulary of labels
        - pos_vocab (dictionary): vocabulary of pos-tags
        - pred_vocab (dictionary): vocabulary of predicates
        - dep_rel_vocab (dictionary): vocabulary of dependency relations
        - lemma_vocab (dictionary): vocabulary of lemmas 
    """
    
    self.file_ = path
    self.labels = {}
    self.sentence = {}
    self.data = []
    self.word2idx = []
    self.list_length = []
    self.tokenizer = tokenizer
    self.sep_tag = "[SEP]"

    self.load_file()

    if w_vocab == None:
      self.w_vocab = self.build_vocab_words()
    else:
      self.w_vocab = w_vocab
    
    if l_vocab == None:
      self.l_vocab = self.build_vocab_labels()
    else:
      self.l_vocab = l_vocab

    if pos_vocab == None:
      self.pos_vocab = self.build_vocab_POS()
    else:
      self.pos_vocab = pos_vocab

    if pred_vocab == None:
      self.pred_vocab = self.build_vocab_predicates()
    else:
      self.pred_vocab = pred_vocab
    
    if dep_rel_vocab == None:
      self.dep_rel_vocab = self.build_vocab_dep()
    else:
      self.dep_rel_vocab = dep_rel_vocab

    if lemma_vocab == None:
      self.lemma_vocab = self.build_vocab_lemma()
    else:
      self.lemma_vocab = lemma_vocab

    self.build_word2idx()
  
  def __getitem__(self, idx):
    return self.word2idx[idx]

  def __len__(self):
    return len(self.word2idx)

  def load_file(self):

    """
      Return: 
        - Initializes the "self.sentence" and "self.labels" lists. Each element of "self.sentence" contains a list of 
          words in the sentence, a list of terms related to the sentence, a list of pos-tags, a list of dependency relations 
          and a list containing a single predicate (list of 0 and 1, 0 if word is not a predicate 1 otherwise). If the sentence 
          contains more than one predicate, it will be repeated as many times as there are predicates. Each element of "self.labels" 
          contains a list of roles associated with that particular predicate in the sentence and predicates.
    """

    with open(self.file_) as f:
      dataset = json.load(f)
    idx = 0
    for i, sentence in dataset.items():
      predicates_sent = self.find_predicates(sentence['predicates'])
      w = sentence['words']
      new_words = []
      for elem in w:
        new_words.append(elem.lower())

      for pred in predicates_sent:
        data = ["_"] *len(w)
        idx_pred = sentence['predicates'].index(pred)
        identify_pred =[0]*len(sentence['words'])
        data[idx_pred] = pred
        identify_pred[idx_pred] = 1
        role = []
        for p, r in sentence['roles'].items():
          p = int(p)
          if p == idx_pred:
            role = r
        self.sentence[idx] = {
            'words': new_words,
            'lemmas': sentence['lemmas'],
            'pos_tags': sentence['pos_tags'],
            'dependency_heads': [int(head) for head in sentence['dependency_heads']],
            'dependency_relations': sentence['dependency_relations'],
            'identify': identify_pred
        }
        
        self.labels[idx] = {
            'predicate': data,
            'roles': role
        }
        item = {"input":self.sentence[idx], "output": self.labels[idx]}
        self.data.append(item)
        idx+=1

      self.list_length.append(len(sentence['words']))

  def find_predicates(self, list_pred):

    """
      Args:
        - list_pred (list): list containing the predicates and not of a sentence

      Return:
        - data : list containing only the predicates of the sentence, "_" are not considered 
    """

    data = []
    for pred in list_pred:
      if pred !="_":
        data.append(pred)
    return data

  def build_vocab_words(self, min_freq=1):

    """
    Args:
      - min_freq: minimum frequency threshold to take a certain word

    Return:
    
      - w_dict: dictionary containing all the words with minimum frequency specified before. 
                In which the key is the word and the value is the integer that represents it.
    """

    w_dict = {}
    counter = Counter()
    for i, elem in tqdm(self.sentence.items()):
      for w in elem["words"]:
        counter[w]+=1
    w_dict.update({'<pad>': 0})
    w_dict.update({'<unk>': 1})
    for index, (key,value) in enumerate(counter.most_common()):
      if value >= min_freq:
        w_dict.update({key: index+2})
    
    return w_dict

  def build_vocab_labels(self):

    """
    Return:
      - l_dict: dictionary containing all the labels (roles). 
                        In which the key is the label and the value 
                        is the integer that represents it.
    """

    l_dict = {}
    counter = Counter()
    for i, elem in tqdm(self.labels.items()):
      for role in elem["roles"]:
        if role is not "_":
          counter[role]+=1
    
    l_dict.update({'_': 1})
    for index, (key,value) in enumerate(counter.most_common()):
      l_dict.update({key: index+2})
    
    return l_dict
  
  def build_vocab_POS(self):

    """
    Return:
      - l_POS: dictionary containing all the pos tags. 
               In which the key is the post tag and the value 
               is the integer that represents it.
    """

    l_POS = {}
    counter = Counter()
    for i, elem in self.sentence.items():
      for pos in elem["pos_tags"]:
        counter[pos]+=1
      
    l_POS.update({'<pad>': 0})

    for index, (key,value) in enumerate(counter.most_common()):
      l_POS.update({key: index+1})
    
    return l_POS
  
  def build_vocab_predicates(self):

    """
    Return:
      - l_pred: dictionary containing all the predicates. 
                In which the key is the predicate and the value 
                is the integer that represents it.
    """

    l_pred= {}
    counter = Counter()
    for i, elem in self.labels.items():
      for pred in elem["predicate"]:
        if pred is not "_":
          counter[pred]+=1
    
    l_pred.update({'<unk>':0})
    l_pred.update({'_': 1})
    for index, (key,value) in enumerate(counter.most_common()):
      l_pred.update({key: index+2})
    
    return l_pred
  
  def build_vocab_dep(self):

    """
    Return:
      - d_dict: dictionary containing all dependency relations. 
                In which the key is the dependency relation and the value 
                is the integer that represents it.
    """

    d_dict = {}
    counter = Counter()
    for i, elem in tqdm(self.sentence.items()):
      for w in elem["dependency_relations"]:
        counter[w]+=1
    d_dict.update({'<pad>': 0})
    for index, (key,value) in enumerate(counter.most_common()):
      d_dict.update({key: index+1})
    
    return d_dict
  
  def build_vocab_lemma(self):

    """
    Return:
      - lem_dict: dictionary containing all dependency relations. 
                In which the key is the dependency relation and the value 
                is the integer that represents it.
    """

    lem_dict = {}
    counter = Counter()
    for i, elem in tqdm(self.sentence.items()):
      for w in elem["lemmas"]:
        counter[w]+=1
    lem_dict.update({'<pad>': 0})
    lem_dict.update({'<unk>': 1})
    for index, (key,value) in enumerate(counter.most_common()):
      lem_dict.update({key: index+2})
    
    return lem_dict
    

  def encode_roles(self, sentence, offset, vocab):

    """
      Args:
        - sentence (list):  list containing the roles of the sentence
        - offset (list of tuple): offset of words caused by BERT tokenization, because a word can be divided into sub-tokens. 
                                  I have to go to align the roles based on these offsets to have a clear correspondence 
                                  between the words and the roles
        - vocab (dict): vocabulary of roles

      Return:
        - data: list of encoding roles aligned according to offset. labels no longer have 
                padding at 0 but at -100, because for simplicity the crossentropy 
                ignores this value. It is also easier to differentiate them from 
                the other elements.
    """

    data = []
    for elem, (start, end) in  zip(sentence,offset):
        data.append(vocab[elem])
        data+=[-100]*(end-start-1)
    return [-100] + data + [-100]
    
  def encode_BERT(self, sentence, offset, vocab):

    """
      Args:
        - sentence (list):  list containing pos-tags or dependency relations of the sentence.
        - offset (list of tuple): offset of words caused by BERT tokenization, because a word can be divided into sub-tokens. 
                                  I have to go to align the elements (pos-tags or dependency relations ) based on these offsets
                                  to have a clear correspondence between the words and the elements.
        - vocab (dict): vocabulary of elements (pos or dependency relations)

      Return:
        - data: list of encoding elements (pos or dependency relations) aligned according to offset. Padding is 0.
    """

    data = []
    for elem, (start, end) in  zip(sentence,offset):
        data.append(vocab[elem])
        data+=[0]*(end-start-1)
    return [0] + data + [0]

  def encode_lemma_BERT(self, sentence, offset, vocab):

    """
      Args:
        - sentence (list):  list containing predicates or lemmas or words (pretrained) of the sentence.
        - offset (list of tuple): offset of words caused by BERT tokenization, because a word can be divided into sub-tokens. 
                                  I have to go to align the elements (predicates or lemmas or pretrained words) based on these offsets
                                  to have a clear correspondence between the words and the elements.
        - vocab (dict): vocabulary of elements (predicates or lemmas or pretrained words)

      Return:
        - data: list of encoding elements (predicate or lemmas or pretrained words) aligned according to offset. If element not in vocabulary we add UNK id.
    """

    data = []
    for elem, (start, end) in  zip(sentence,offset):
      if elem in vocab.keys():
        data.append(vocab[elem])
      else:
        data.append(vocab["<unk>"])
      data+=[0]*(end-start-1)
    return [0] + data + [0]
 
  def encode_pred_BERT(self, sentence, offset, vocab):

    """
      Args:
        - sentence (list):  list containing the predicates of the sentence
        - offset (list of tuple): offset of words caused by BERT tokenization, because a word can be divided into sub-tokens. 
                                  I have to go to align the predicates based on these offsets to have a clear correspondence 
                                  between the words and the predicates
        - vocab (dict): vocabulary of predicates

      Return:
        - data: list of encoding predicates aligned according to offset. labels no longer have 
                padding at 0 but at -100, because for simplicity the crossentropy 
                ignores this value. It is also easier to differentiate them from 
                the other elements.
    """

    data = []
    for elem, (start, end) in  zip(sentence,offset):
      if elem in vocab.keys():
        data.append(vocab[elem])
      else:
        data.append(vocab["<unk>"])
      data+=[-100]*(end-start-1)
    return [-100] + data + [-100]

  def encode_idx_BERT(self, sentence, offset):

    """
      Args:
        - sentence (list):  list containing 0 and 1 to identify predicates in the sententece.
        - offset (list of tuple): offset of words caused by BERT tokenization, because a word can be divided into sub-tokens. 
                                  I have to go to align element of list based on these offsets
                                  to have a clear correspondence between the words and the elements.

      Return:
        - data: list of encoding elements 0 and 1 aligned according to offset. Padding is 0.
    """

    data = []
    for elem, (start, end) in  zip(sentence,offset):
        data.append(elem)
        data+=[0]*(end-start-1)
    return [0] + data + [0]

  def build_word2idx(self):

    """
      Return:
        - Initialize vector word2idx. 
          Each element is a tuple contains:
            - sub_word_idx: encoding sentence following BertTokenizer
            - lemma: encoding lemmas with offset
            - pos: encoding pos-tags with offset
            - ident: encoding identify predicates with offset
            - dep : encoding dependency relations with offset
            - pred: encoding predicates with offset
            - lab: encoding labels with offset
            - data: adjacent matrix of sentence (A+I)
            - degree: matrix to normalize adjacent matrix (D^(-1/2))
            - word_pretrained: encoding pretrained words
    """

    for elem in tqdm(self.data):
      inp = elem["input"]
      out = elem["output"]

      sub_word_idx = []
      word_offset = [] # very important
      curr_offset = 1

      # words

      words = inp["words"]
      
      for w in words:

        word_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(w))
        sub_word_idx += word_idx
        word_offset.append((curr_offset, curr_offset+ len(word_idx)))  # use to understand offset for subwords
        curr_offset +=len(word_idx) # take care of offset

      sub_word_idx = self.tokenizer.build_inputs_with_special_tokens(sub_word_idx) 

      word_pretrained = self.encode_lemma_BERT(words, word_offset, self.w_vocab)
      # labels 

      role = out["roles"]

      labels = self.encode_roles(role,word_offset, self.l_vocab)

      # lemmas

      lemmas = inp["lemmas"]
      lemma = self.encode_lemma_BERT(lemmas, word_offset, self.lemma_vocab)

      # identify
      identify = inp['identify']
      ident = self.encode_idx_BERT(identify,word_offset)

      
      # POS tag

      POS = inp["pos_tags"]
      pos = self.encode_BERT(POS,word_offset, self.pos_vocab)

      # Predicate

      predic = out["predicate"]
      pred = self.encode_pred_BERT(predic, word_offset, self.pred_vocab)      

      # dependency
      
      dependency = inp["dependency_relations"]
      dep = self.encode_BERT(dependency,word_offset,self.dep_rel_vocab)

      # dependency heads

      dep_heads = inp["dependency_heads"]
      data = torch.zeros([len(sub_word_idx), len(sub_word_idx)])
      h_d = []
      for idx, elem in enumerate(dep_heads): 
        if idx == elem:
          pass
        h_d.append((idx,elem))
        h_d.append((elem,idx))

      for idx,elem in h_d:
        for i in range(len(data)):
          if i == idx:
            row = data[i]
            for j in range(len(row)):
              if j == elem-1:
                row[j] = 1
      
      identity_M = torch.eye(len(sub_word_idx))
      degree_tokens = []
      data = torch.add(data,identity_M)
      
      for i in range(len(data)):
        sum_tokens = 0
        row = data[i]
        for j in range(len(row)):
          sum_tokens+=row[j]
          degree_tokens.append(sum_tokens)
      
      degree = torch.eye(len(sub_word_idx))
      for i in range(len(degree)):
        for j in range(len(degree[i])):
          if i == j:
            degree[i,j] = degree_tokens[i]
      degree = degree.numpy() 
      degree = fractional_matrix_power(degree, -0.5)
      degree = torch.from_numpy(degree)


      item =(
           torch.tensor(sub_word_idx),
           torch.tensor(lemma),
           torch.tensor(pos),
           torch.tensor(ident),
           torch.tensor(dep),
           torch.tensor(pred),
           torch.tensor(labels),
           data,
           degree,
           torch.tensor(word_pretrained),
      )

      self.word2idx.append(item)
