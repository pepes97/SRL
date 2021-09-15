import torch
from torch import nn
from transformers import BertModel
from torch.nn.parameter import Parameter

class SvevaModelBERT_GCN_DIS(nn.Module):
    def __init__(self, hparams):
        super(SvevaModelBERT_GCN_DIS, self).__init__()

        # Embedding layer: a matrix vocab_size x embedding_dim where each index 
        # correspond to a word in the vocabulary and the i-th row corresponds to 
        # a latent representation of the i-th word in the vocabulary.

        # 6 embeddings: BERT + Glove + pos + identify predicates + dependency relations + lemma

        self.word_embedding = BertModel.from_pretrained(hparams.transformer_model)
        self.identify_embedding = nn.Embedding(hparams.pred_vocab_size, hparams.embedding_dim_pred)
        self.POS_embedding = nn.Embedding(hparams.POS_vocab_size, hparams.embedding_dim_POS)
        self.dep_embedding = nn.Embedding(hparams.dep_rel_vocab_size, hparams.embedding_dim_dep_rel)
        self.lemma_embedding = nn.Embedding(hparams.lemma_vocab_size, hparams.embedding_dim_lemma)
        self.pretrained_embedding = nn.Embedding(hparams.w_vocab_size, hparams.embedding_dim_pretrained)

        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.pretrained_embedding.weight.data.copy_(hparams.embeddings.weights)

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words.

        self.lstm = nn.LSTM(hparams.embedding_dim_words+
                            hparams.embedding_dim_POS+
                            hparams.embedding_dim_pred+
                            hparams.embedding_dim_dep_rel+
                            hparams.embedding_dim_lemma+
                            hparams.embedding_dim_pretrained,
                            hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            batch_first=True,
                            dropout = hparams.dropout if hparams.num_layers > 1 else 0)
        
        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.

        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2

        # weight and bias for GCN

        self.weight = Parameter(torch.randn(hparams.batch_size,lstm_output_dim, lstm_output_dim))
        self.bias = Parameter(torch.randn(hparams.batch_size,1, lstm_output_dim))

        # RELU + sigmoid

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # During training, randomly zeroes some of the elements of the 
        # input tensor with probability hparams.dropout using samples 
        # from a Bernoulli distribution. Each channel will be zeroed out 
        # independently on every forward call.
        # This has proven to be an effective technique for regularization and 
        # preventing the co-adaptation of neurons, so to avoid overfitting

        self.dropout = nn.Dropout(hparams.dropout)
        self.pred_class = nn.Linear(lstm_output_dim, hparams.num_pred)
        self.classifier = nn.Linear(lstm_output_dim+lstm_output_dim+hparams.pred_dim, hparams.num_classes)

        if not hparams.fine_tune:
            for param in self.word_embedding.parameters():
                param.requires_grad = False 

    
    def forward(self,x,y,w,pretrained,idx,k,adj,degree,mask):
        embeddings_word = self.word_embedding(x, attention_mask=mask)[0]
        embeddings = self.dropout(embeddings_word)
        embeddings_POS = self.POS_embedding(y)
        embeddings_dep = self.dep_embedding(w)
        embeddings_lemma = self.lemma_embedding(k)
        embeddings_idx = self.identify_embedding(idx)
        embeddings_pretrained = self.pretrained_embedding(pretrained)
        embeddings = torch.cat((embeddings,embeddings_idx, embeddings_POS, embeddings_dep, embeddings_lemma,embeddings_pretrained), dim=2)
        o, (h, c) = self.lstm(embeddings)
        out = o 
        if o.shape[0] != self.weight.shape[0]:
          out = torch.bmm(out, self.weight[:o.shape[0]])
          out = torch.add(out, self.bias[:o.shape[0]])
        else:
          out = torch.bmm(out, self.weight)
          out = torch.add(out, self.bias)
        prod = torch.bmm(degree,adj)
        prod = torch.bmm(prod,degree)
        prod = torch.bmm(prod,out)
        relu = self.relu(prod)
        #sigmoid = self.sigmoid(relu)
        o = self.dropout(o)
        pred = self.pred_class(o)
        tot = torch.cat((o,pred,relu), dim=2)
        output = self.classifier(tot)
        return pred,output


class HParamsBERT_GCN_DIS():
    def __init__(self,w_vocab, l_vocab, pos_vocab, pred_vocab, 
                       dep_vocab,lemma_vocab, bert_model, 
                       embeddings, embedding_dim_word,
                       embedding_dim_pretrained, embedding_dim_POS,
                       embedding_dim_pred, embedding_dim_dep_rel,
                       embedding_dim_lemma, batch_size, 
                       hidden_dim, bidirectional,
                       num_layers, dropout, pred_dim):
  
      self.transformer_model = bert_model
      self.POS_vocab_size = len(pos_vocab)
      
      self.w_vocab_size = len(w_vocab)
      self.dep_rel_vocab_size = len(dep_vocab)
      self.lemma_vocab_size = len(lemma_vocab)
      self.fine_tune = False

      self.embedding_dim_words = embedding_dim_word
      self.embedding_dim_pretrained = embedding_dim_pretrained
      self.embedding_dim_POS = embedding_dim_POS
      self.embedding_dim_pred = embedding_dim_pred
      self.embedding_dim_dep_rel = embedding_dim_dep_rel
      self.embedding_dim_lemma = embedding_dim_lemma
      self.batch_size=batch_size
      
      
      self.hidden_dim = hidden_dim
      self.num_classes = len(l_vocab)
      self.bidirectional = bidirectional
      self.num_layers = num_layers
      self.dropout = dropout
      self.embeddings = embeddings

      self.pred_vocab_size= 2
      self.pred_dim = pred_dim
      self.num_classes = len(l_vocab)
      self.num_pred = len(pred_vocab)