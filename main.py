
from lib.model_dis import HParamsBERT_GCN_DIS, SvevaModelBERT_GCN_DIS
import torch
import os
import argparse 
from torch import nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from lib.dataset import SvevaDatasetBERT_GCN
from lib.dataset_dis import SvevaDatasetBERT_GCN_DIS
from lib.utils import padding_function_BERT_GCN, compute_metrics, compute_metrics_dis
from lib.pretrained import PreTrainedEmbedding
from lib.model import SvevaModelBERT_GCN, HParamsBERT_GCN
from lib.trainer import TrainerBERT_GCN
from lib.trainer_dis import TrainerBERT_GCN_DIS
from lib.model_dis import SvevaModelBERT_GCN_DIS, HParamsBERT_GCN_DIS

SEED = 1234
np.random.seed(SEED)

def main(type_bert, batch_size, embedding_dim_word, embedding_dim_pretrained, embedding_dim_pos, embedding_dim_pred, embedding_dim_dep_rel, embedding_dim_lemma, hidden_dim, bidirectional, num_layers, dropout, learning_rate, epochs, only_test, pred_disamb, pred_dim):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\033[1mDevice \033[0m: {device} \033[0m")

    train_file = "./dataset/train.json"
    dev_file = "./dataset/dev.json"
    test_file = "./dataset/test.json"
    glove_file = "./dataset/glove.6B.50d.txt"

    print(f"\033[1mTrain file \033[0m: {train_file} \033[0m")
    print(f"\033[1mDev file \033[0m: {dev_file} \033[0m")
    print(f"\033[1mTest file \033[0m: {test_file} \033[0m")

    print(f"\033[1mType Bert: {type_bert} \033[0m")
    bert_model = "bert-"+type_bert+"-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    print(f"\033[1mCreating dataset... \033[0m")
    
    if not pred_disamb:
        trainset = SvevaDatasetBERT_GCN(train_file,tokenizer)
        words_vocabulary = trainset.w_vocab
        label_vocabulary = trainset.l_vocab
        POS_vocabulary = trainset.pos_vocab
        pred_vocabulary = trainset.pred_vocab
        dep_relation_vocabulary = trainset.dep_rel_vocab
        lemma_vocabulary = trainset.lemma_vocab
        devset = SvevaDatasetBERT_GCN(dev_file, tokenizer,words_vocabulary,label_vocabulary, POS_vocabulary, 
                        pred_vocabulary,dep_relation_vocabulary,lemma_vocabulary)

        testset = SvevaDatasetBERT_GCN(test_file,tokenizer,words_vocabulary,label_vocabulary, POS_vocabulary, 
                        pred_vocabulary,dep_relation_vocabulary, lemma_vocabulary)
    
    else:
        trainset = SvevaDatasetBERT_GCN_DIS(train_file,tokenizer)
        words_vocabulary = trainset.w_vocab
        label_vocabulary = trainset.l_vocab
        POS_vocabulary = trainset.pos_vocab
        pred_vocabulary = trainset.pred_vocab
        dep_relation_vocabulary = trainset.dep_rel_vocab
        lemma_vocabulary = trainset.lemma_vocab
        devset = SvevaDatasetBERT_GCN_DIS(dev_file, tokenizer,words_vocabulary,label_vocabulary, POS_vocabulary, 
                        pred_vocabulary,dep_relation_vocabulary,lemma_vocabulary)

        testset = SvevaDatasetBERT_GCN_DIS(test_file,tokenizer,words_vocabulary,label_vocabulary, POS_vocabulary, 
                        pred_vocabulary,dep_relation_vocabulary, lemma_vocabulary)
    
    
    train_dataset =DataLoader(trainset, batch_size=batch_size, collate_fn=padding_function_BERT_GCN, shuffle = True)
    valid_dataset = DataLoader(devset, batch_size=batch_size, collate_fn=padding_function_BERT_GCN)
    test_dataset = DataLoader(testset, batch_size=batch_size, collate_fn=padding_function_BERT_GCN)

    pre_embeddings = PreTrainedEmbedding(glove_file,50,words_vocabulary)

    if not pred_disamb:
        paramsBERT_GCN = HParamsBERT_GCN(words_vocabulary, 
                                        label_vocabulary, 
                                        POS_vocabulary, 
                                        pred_vocabulary, 
                                        dep_relation_vocabulary, 
                                        lemma_vocabulary, 
                                        bert_model, pre_embeddings, 
                                        embedding_dim_word,
                                        embedding_dim_pretrained, 
                                        embedding_dim_pos,
                                        embedding_dim_pred, embedding_dim_dep_rel,
                                        embedding_dim_lemma, batch_size, 
                                        hidden_dim, bidirectional,
                                        num_layers, dropout)
        
        modelBERT_GCN = SvevaModelBERT_GCN(paramsBERT_GCN).to(device)

    else:
        paramsBERT_GCN = HParamsBERT_GCN_DIS(words_vocabulary, 
                                        label_vocabulary, 
                                        POS_vocabulary, 
                                        pred_vocabulary, 
                                        dep_relation_vocabulary, 
                                        lemma_vocabulary, 
                                        bert_model, pre_embeddings, 
                                        embedding_dim_word,
                                        embedding_dim_pretrained, 
                                        embedding_dim_pos,
                                        embedding_dim_pred, embedding_dim_dep_rel,
                                        embedding_dim_lemma, batch_size, 
                                        hidden_dim, bidirectional,
                                        num_layers, dropout, 
                                        pred_dim)
        
        modelBERT_GCN = SvevaModelBERT_GCN_DIS(paramsBERT_GCN).to(device)


    if not pred_disamb:
        trainer = TrainerBERT_GCN(
            model = modelBERT_GCN,
            loss_function = nn.CrossEntropyLoss(),
            optimizer = optim.Adam(modelBERT_GCN.parameters(),lr=learning_rate),
            tokenizer = tokenizer,
            device = device,
            label_vocab= label_vocabulary 
        )
    else:
        trainer = TrainerBERT_GCN_DIS(
            model = modelBERT_GCN,
            loss_function = nn.CrossEntropyLoss(),
            optimizer = optim.Adam(modelBERT_GCN.parameters(),lr=learning_rate),
            tokenizer = tokenizer,
            device = device,
            label_vocab= label_vocabulary 
        )
    
    if not only_test:
        if not os.path.exists("./models"):
            os.mkdir("./models")
        print(f"\033[1mTraining... \033[0m")
        trainer.train(train_dataset,valid_dataset,epochs)
    else:
        if not pred_disamb:
            if os.path.exists('models/best_model.pt'):
                print(f"\033[1mLoad Best Model... \033[0m")
                trainer.model.load_state_dict(torch.load('models/best_model.pt'), map_location=device)
            else:
                print(f"\033[1mBest Model not found... \033[0m")
                exit()
        else:
            if os.path.exists('models/best_model_dis.pt'):
                print(f"\033[1mLoad Best Model DIS... \033[0m")
                trainer.model.load_state_dict(torch.load('models/best_model.pt'), map_location=device)
            else:
                print(f"\033[1mBest Model not found... \033[0m")
                exit()

    if not pred_disamb:
        print(f"\033[1mTesting... \033[0m")
        precisions = compute_metrics(modelBERT_GCN, test_dataset, label_vocabulary, tokenizer, device)
        mp = precisions["macro_precision"]
        r = precisions["recall"]
        f1= precisions["f1"]
        print(f"\033[1mMacro Precision: {mp} \nRecall: {r} \nF1_score: {f1}\033[0m")
    
    else:
        print(f"\033[1mTesting... \033[0m")
        precisions = compute_metrics_dis(modelBERT_GCN, test_dataset, label_vocabulary, tokenizer, device)
        mp = precisions["macro_precision"]
        r = precisions["recall"]
        f1= precisions["f1"]
        print(f"\033[1mMacro Precision: {mp} \nRecall: {r} \nF1_score: {f1}\033[0m")
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--type-bert", type=str, default="base", help="base or large of Bert")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument('--embedding-dim-word', type=int, default=768, help='dimension of word embedding')
    parser.add_argument('--embedding-dim-pretrained', type=int, default=50, help='dimension of pretrained word embedding')
    parser.add_argument('--embedding-dim-pos', type=int, default=32, help='dimension POS embedding')
    parser.add_argument('--embedding-dim-pred', type=int, default=50, help='dimension pred embedding')
    parser.add_argument('--embedding-dim-dep-rel', type=int, default=50, help='dimension dependency relations embedding')
    parser.add_argument('--embedding-dim-lemma', type=int, default=50, help='dimension lemma embedding')
    parser.add_argument('--hidden-dim', type=int, default=256, help='dimension hidden layer LSTM')
    parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional LSTM')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--only-test', type=bool, default=False, help='only testing')
    parser.add_argument('--pred-disamb', type=bool, default=False, help='flag for addition of predicate dimbiguation')
    parser.add_argument('--pred-dim', type=int, default=457, help='dimension pred')
    
    args = parser.parse_args()
    type_bert = args.type_bert
    batch_size = args.batch_size
    embedding_dim_word = args.embedding_dim_word
    embedding_dim_pretrained = args.embedding_dim_pretrained
    embedding_dim_pos = args.embedding_dim_pos
    embedding_dim_pred = args.embedding_dim_pred
    embedding_dim_dep_rel = args.embedding_dim_dep_rel
    embedding_dim_lemma = args.embedding_dim_lemma
    hidden_dim = args.hidden_dim
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    dropout = args.dropout
    learning_rate = args.lr
    epochs = args.epochs
    only_test = args.only_test
    pred_disamb= args.pred_disamb
    pred_dim = args.pred_dim
    
    main(type_bert, batch_size, embedding_dim_word, embedding_dim_pretrained, embedding_dim_pos, embedding_dim_pred, embedding_dim_dep_rel, embedding_dim_lemma, hidden_dim, bidirectional, num_layers, dropout, learning_rate, epochs, only_test, pred_disamb, pred_dim)
