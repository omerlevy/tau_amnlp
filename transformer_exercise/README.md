# Modifying the Transformer Architecture

**Huge thanks to Tomer Amit for coding, testing, and organizing this exercise!**

This exercise aims to familiarize you with a real-world implementation of the transformer.
We will use the popular **fairseq** platform, which is designed for experimenting with machine translation, language modeling, pre-training, and other sequence processing tasks.
You will implement some of the experiments described in Lesson 3 by modifying the transformer's code.

## Installation

_TODO: Check which of these we actually need to install. I don't think we need to install fairseq, for example, because we're alreayd using a clone._

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```

To install fairseq and pandas:
```bash
pip install fairseq
pip install pandas
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
    --save-dir baseline
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --best-checkpoint-metric ppl 
    --maximize-best-checkpoint-metric \
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

In this part of the excersice, we will see the effect of masking different heads in the transformer layers.

```
CUDA_VISIBLE_DEVICES=0 python generate.py data-bin/iwslt14.tokenized.de-en \
    --path baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
	--fp16 \
    --model-overrides "{'mask_layer': 5, 'mask_head': 3, 'mask_layer_name': 'enc-dec'}"
```
mask_layer is the layer number to mask
mask_head is the head number to mask
mask_layer_name is the name of the attention to mask - 'enc-enc' is the transformer encoder self attention
													 - 'enc-dec' is the transformer decoder cross attention
													 - 'dec-dec' is the transformer decoder self attention

follow this arguments to see their impact.

in the end, the mask_head argument, turn into head_to_mask variable on the function forward in the multihead_attention.py file.
your task is to implement the mask part inside the forward function (1-3 lines)

finally after everything is ready, execute check_all_masking_options.py that execute mask of each attention head in each transformer to see it's impact

```
CUDA_VISIBLE_DEVICES=0 python check_all_masking_options.py data-bin/iwslt14.tokenized.de-en --path baseline/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --fp16
```

validate your results and explain them.

## Training a sandwitch model on IWSLT'14 German to English

In this part of the excersice, we will see the effect of different configuration of the transformers.
As mentioned in lecture 3, both the encoder transformer and the decoder transformer contain Multi head attention part - MHA (in the decoder we reffer to
both self attention and cross attention) followed by 2 layer feed forward netowork - FFN

we reffer the regular configuration as AFAFAFAFAFAF (we mark MHA layer as A and the FFN part layer as F)
and now we will check another configuration, and see it's result

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --save-dir baseline
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 12288 \
    --best-checkpoint-metric ppl 
    --maximize-best-checkpoint-metric \
    --fp16 \
    --enc-layer-configuration 'FFFFFFAAAAAA'
    --dec-layer-configuration 'FFFFFFAAAAAA'
```

follow the enc-layer-configuration and dec-layer-configuration arguments and implement TransformerEncoderLayerFFN TransformerEncoderLayerMHA 
TransformerDecoderLayerMHA and TransformerEncoderLayerFFN
