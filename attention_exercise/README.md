# Attention 

This exercise is the based on the attention models discussed in lecture two of the course. 

We will work on an annotated dataset called SemCor https://www.gabormelli.com/RKB/SemCor_Corpus, 
and implement a few variations of the basic attention model following to the exercises described in the lecture.

The dataset is comprised of a collection of sentences, each of them we will consider as seperate context.
In every sentence, all words with clear sense were labeled, resulting in a total of
roughly 30% annotated words. The 70% remainder of the words were given a "no_sense" label.
To describe a word in a sentence, we may use the terms word or token interchangeably (my apologies).

You will need a gpu to train the model, google collab should be a good choice. 
Remember to upload the three python files data_loader.py, model.py, and traineval.py for the notebook 
to be able to use them.

## Background
The SemCor dataset was pre-processed and split to train (80%), dev (10%) and test(10%) using the build_dataset.py script.

**Question for Omer** --> What is the proper way to handle the test set? Since all datasets should share the vocabulary,
should we just load them all together from the beginning?

All the code you will need to implement will either be in model.py, or in the wsd_model.ipynb notebook 
(abbreviated hereafter as **the** notebook). That said, feel free to change other parts of the code as well if you like. 

It is recommended to develop and debug locally, and only use collab for the full datasets once you have 
a properly working model. Alternatively, you may choose to copy the model code into collab notebook, thus eliminating
the need to upload a file after every code change.

If you look at data_loader.py, you will see 3 classes:
* Vocabulary implementation, which serves for vocabulary bookkeeping of sense and word integer ids encoding and decoding.
* WSDDataset serves single word attention, in which a sample consists of the 3-tuple (sentence, query, label),
The query is the index of the to-be-disambiguated word in the sentence. The label is the annotated sense.
All three tensors are integer encoded, and their string representation can be looked uo in the appropriate vocabulary.
Note there is a special key for non labeled words - 'no_sense'. These samples are not served by this dataset
implementation.
* WSDSentencesDataset serves self attention, in which a sample consists of the 2-tuple (sentence, labels),
In this case, the sentence itself serves as the query, and the labels provide annotated sense for every word in the 
sentence.

Note that in order to facilitate batching, the sentences (in both dataset implementations) and labels (in the 2nd one) 
are padded to the maximum sentence length.

We will follow notation used in the lecture for the most part, and use the following dimension variables;
* B: Batch dimension, will be 100 unless you change it.
* N: Sentence length, automatically computed by WSDDataset according to the longest sentence.
* D: Embedding dimension, set to 300 by default.

The notebook will walk you through the train / eval / analyse process for Exercise 2A.
Exercises 2B and 2C should follow the same basic workflow.


## Exercise 2A: Attention

###  2A.1: Single Query Attention
Take a look at model.py. The WSD model is already initialized with embeddings and attention matrices with variable names 
following notation from the lecture. Fill in your code at the "TODO Ex2A" placeholders.
The v_q argument representing the query is optional, since we will use the same model to implement self attention.
Don't worry for now about the mask argument passed to the attention function - we'll deal with that later. 

Single word attention for word q in context X is given by
```
a = softmax(q @ W_A @ X.T)
q_c = a @ X @ W_O
```

where @ denotes matrix multiplication, q - the query word embedding, W_A, W_O attention parameters matrices,
and X the context embedding - sentence in out case.
The prediction produced by the model is then computed as
```
h = layer_norm(q_c + q)
p(y) = softmax(h @ E_Y.T)
```

Use the notebook and the hyperparameters as specified by it to train your model, and plot loss and accuracy to validate convergence.  
When things are looking sharp, you can proceed to visualize the model's attention using the api provided as demonstrated.

Can you notice anything that might be off? 

###  2A.2: Attending Padding
Apparently, the model learns to "attend" the padded indices as if they were legitimate tokens.
As the padding was introduced to solve a technical problem and should not be considered by the model at all, 
you are going to mask it out.

Use the mask argument in your freshly implemented attention function to "zero out" any padding attention
factors. Note that to zero out a softmax output., you should provide -inf in appropriate locations in the input tensor.

### 2A.3: Self Attention
Change your implementation so that it can handle self attention as described in the lecture.

Recall that in self attention mode, the model is served with the sentences matrix (M_s) only, i.e. the optional
v_q argument is not passed. Take a look at the training / eval implementation - the logic there switches between 
the two modes according to a flag on the dataset object; sample_type='word' or sample_type='sentence'.
The notebook will take you through the simple process of converting the word level datasets to
sentence level ones.

Self attention for context X is given by
```
A = softmax(X @ W_A @ X.T)
Q_c = A @ X @ W_O
```

Follow the notebook to run training, loss inspection and visualization of attention.
Notice the change in performance, and the earlier convergence.

## Exercise 2B: Position-Sensitive Attention
Extend your model to add position sensitivity.
Verify your newly attained position sensitivity using the attention visualization provided.


## Exercise 2D: Causal Attention
Implement a causal attention model, by extending your existing one with a "causal mode".
Verify causality using the attention visualization provided.
 
 
 **NOTE**: Depending on implementation, highlight visualization code might need to be changed to accomodate 2B and 2C, 
 as it currently supports only models that can predict single word attention.
 