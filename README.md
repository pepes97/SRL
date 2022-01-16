# Semantic Role Labeling

Semantic Role Labeling is a fundamental NLPtask, which has the goal of finding semantic roles for each predicate in a sentence.
The goal of the SRL is to extract predicate-argument structure of a sentence, identifying ”who did what to whom”, ”when”, ”where” etc.For example, consider this sentence:The cat eats a fish. Eats is the verb,The cat is the subject and a fish is the object complement. We are not interested in the meaning of ”cat” or ”fish”, but we want to identify and classify them, i.e, associate each argument with its corresponding role. To solve this task, LSTM-based models in different configurations were used in this paper, including pre-trained word embeddings, contextualized word embedding from BERT and Graph Convolutional Network. Furthermore, the subtask of disambiguation of predicates is also taken into consideration, because every often the datasets that are provided have information on the predicates present in the sentences, but not the clarification of the meaning of them.

## Table of contents

* [Development Setup](#Development-Setup)
* [Requirements](#Requirements)
  * [Download the glove data](#Download-Dataset)
  * [Run](#Run)

### Development Setup

<p align="center">
  <img width="600" height="350" src="https://user-images.githubusercontent.com/56698309/149672469-c53f4a08-5297-487d-a076-9cbddc242818.png">
</p>

### Requirements 

```
conda create --name srl python=3.7
conda activate srl
pip install -r requirements.txt
```
### Download Dataset
```
$ cd dataset
$ python download_dataset.py
```

### Run

```
python main.py type-bert TYPE-BERT --batch-size BATCH-SIZE --embedding-dim-word EMBEDDING-DIM-WORD --embedding-dim-pretrained -- EMBEDDING-DIM-PRETRAINED --embedding-dim-pos EMBEDDING-DIM-POS --embedding-dim-pred EMBEDDING-DIM-PRED embedding-dim-dep-rel EMBEDDING-DIM-DEP-REL --embedding-dim-lemma EMBEDDING-DIM-LEMMA --hidden-dim HIDDEN-DIM --epochs EPOCHS --batch-size BATCH-SIZE --lr LEARNING-RATE --dropout DROPOUT --bidirectional BIDIRECTIONAL --num-layers NUM-LAYERS --only-test ONLY-TEST --pred-disamb PRED-DISAMB --pred-dim PRED-DIM
```
where

- `TYPE-BERT` type of BERT: base or large, default is base
- `EMBEDDING-DIM-WORD` is the dimension of the word embedding, default is 768
- `EMBEDDING-DIM-PRETRAINED` is the dimension of the word pretrained embedding, default is 50
- `EMBEDDING-DIM-POS` is the dimension of pos embedding, default is 32
- `EMBEDDING-DIM-PRED` is the dimension of predicate embedding, default is 50
- `EMBEDDING-DIM-DEP-REL` is the dimension of dependency relations embedding, default is 50
- `EMBEDDING-DIM-LEMMA` is the dimension of lemma embedding, default is 50
- `HIDDEN-DIM` is the dimension of the hidden layer, default is 256
- `EPOCHS` is the number of epochs, default is 15
- `BATCH-SIZE` is the batch size, default is 64
- `LEARNING-RATE` is the learning rate, default is 0.001
- `DROPOUT` is the dropout rate, default is 0.5
- `BIDIRECTIONAL` is the bidirectional flag, default is True
- `NUM-LAYERS` is the number of layers, default is 2
- `ONLY-TEST` is the only test flag, default is False
- `PRED-DISAMB` is the flag for addition of predicate dimbiguation, default is False
- `PRED-DIM` is the dimension of the predicates, default is 457

for example, for training:

```
python main.py
```

for testing
```
python main.py --only-test True
```
