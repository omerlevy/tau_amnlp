# Modifying the Transformer Architecture

**Huge thanks to Tomer Amit for coding, testing, and organizing this exercise!**

This exercise aims to familiarize you with a real-world implementation of the transformer.
We will use the popular **fairseq** platform, which is designed for experimenting with machine translation, language modeling, pre-training, and other sequence processing tasks.
You will implement some of the experiments described in Lesson 3 by modifying the transformer's code.

## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

To install pandas:
```bash
pip install pandas
```

Finally, to install this version of fairseq:
```bash
pip install --editable .
```

## Part 1: Training a Machine Translation Model

We will first train a baseline machine translation model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).
This dataset is made from subtitles of TED talks in many different languages. Specifically, we recommend using the popular German to English subset, but you may also experiment with Hebrew/Arabic/Russian/etc by changing the data downloading script.

**Step 1:** Download and preprocess the data.
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

**Step 2:** Train an encoder-decoder transformer using this data.
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir baseline \
    --max-epoch 50 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu \
    --fp16
```

* The specific configuration we will be using (```--arch transformer_iwslt_de_en```) has 6 encoder/decoder layers and 4 attention heads for each multi-head attention sublayer. This amounts to 24 attention heads of each type (enc-enc, enc-dec, dec-dec), which are 72 heads overall.

* The ```--save-dir baseline``` argument will save the model into the "baseline" folder.

* If your GPU runs out of memory, you can reduce ```--max-tokens```.

* ```--fp16``` Makes training faster by using mixed floating-point precision. It has very little effect on the performance, so if your GPU is a bit older and does not support it, do not worry.

**Step 3:** Evaluate the trained model.
```
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en --path baseline/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --fp16
```

This step runs beam search (Lesson 5) and generates discrete strings. The system's outputs are compared to the reference strings in the test set using BLEU. If you have trained the model for long enough (50 epochs), you should be getting a BLEU score of about 34 to 35.


## Part 2: Masking Attention Heads

We would like to replicate one of the experiments in [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650), in which we mask a single attention head each time and test the model's performance.
To that end, we have added an additional command-line argument to ```generate.py``` that specifies which attention head needs to be masked (see below). Your task is to mask the correct head given the argument.

Specifically, you should read the command line arguments (```args.mask_layer```, ```args.mask_head```, ```args.mask_layer_type```) in the transformer's constructor and pass a masking flag to the relevant multi-head attention sublayer. The transformer's implementation can be found in [fairseq/models/transformer.py](fairseq/models/transformer.py). Note that there are two classes (encoder and decoder), and both need to be modified.

You will then need to implement the actual masking in [fairseq/modules/multihead_attention.py](fairseq/modules/multihead_attention.py).

After making your changes, you can debug them by running the following script:
```
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    --path baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
	--fp16 \
    --model-overrides "{'mask_layer': 5, 'mask_head': 3, 'mask_layer_type': 'enc-dec'}"
```
The arguments ```mask_layer``` and ```mask_head``` specify the layer (0-5) and head (0-3) to mask, while ```mask_layer_type``` specifies which attention type is being masked ('enc-enc', 'enc-dec', 'dec-dec'). If your code works correctly, you should see a reduction in performance of about 1-3 BLEU.

When you are done implementing and testing your code, execute [check_all_masking_options.py](check_all_masking_options.py) as follows:
```
CUDA_VISIBLE_DEVICES=0 python check_all_masking_options.py data-bin/iwslt14.tokenized.de-en --path baseline/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --fp16
```
This will run the same masking experiment for each attention head in the model, and print out result tables.

What do we learn from these results? Which heads can be safely removed without incurring a significant change in performance (up to 0.5 BLEU)? Which heads are more critical? Is the trend consistent with the findings in the paper?


## Part 3: Reordering Transformer Sublayers

We would like to replicate the idea in [Improving Transformer Models by Reordering their Sublayers](https://arxiv.org/abs/1911.03864), where new transformer configurations can be created by chaining multi-head attention and feed-forward sublayers in arbitrary patterns. Your task is to implement this ability in fairseq.

We have added the ```enc-layer-configuration``` and ```dec-layer-configuration``` arguments to fairseq. These receive a string made of "A"s and "F"s, symbolizing multi-head attention feed-forward sublayers, respectively. For example, the original transformer architecture is "AFAFAFAFAFAF". Note that in the decoder, we define "A" as (causal) self-attention followed by cross attention and treat both sublayers as a single atomic unit.

To test your implementation, you will have to retrain the transformer. Here is an example of one architecture in which all the feed-forward layers are applied before the multi-head attention sublayers:
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir sandwich \
    --max-epoch 50 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --eval-bleu \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu \
    --fp16 \
    --enc-layer-configuration 'FFFFFFAAAAAA' \
    --dec-layer-configuration 'FFFFFFAAAAAA'
```

Once you are done implementing and testing your code, train two additional configurations of your choosing, and determine (by evaluating them with ```generate.py```, as in Part 1) whether your proposed modification improved, hurt, or did not have a significant effect on performance. Here are some ideas for possible patterns:
* Sandwich transformers ("AAAFAFAFAFFF")
* No feed-forward layers at all ("AAAAAA" or "AAAAAAAAAAAAAAAAAA", the latter being equivalent to the baseline in number of parameters)
* Less attention, more feed-forward ("AFFFAFFF" or "AFFFAFFFFF", the latter being equivalent to the baseline in number of parameters)

**Good luck!**
