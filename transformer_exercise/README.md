# Modifying the Transformer Architecture

This exercise aims to familiarize you with a real-world implementation of the transformer.
We will use the popular **fairseq** platform, which is designed for experimenting with machine translation, language modeling, pre-training, and other sequence processing tasks.
You will implement some of the experiments described in class by modifying the transformer's code.

## Submission:

Please submit a zip file containing the following files and logs that you will get from your runs:
* baseline_train.log
* baseline_gen.log
* baseline_mask.log
* check_all_masking_options.log
* sandwich_train.log
* sandwich_gen.log
* transformer_exercise/fairseq/models/transformer.py
* transformer_exercise/fairseq/modules/transformer_layer.py
* transformer_exercise/fairseq/modules/multihead_attention.py
* any additional file that you choose to change 
* summary.md

Please do not forget to fill and submit the summary.md file!

Only one student from each group should submit.

Name the zip file with your IDs separated by underscores.
For example: 123_456.zip

## Installation

Requirements:
* A GPU (or Colab)
* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.4.0

Install the required libraries:
```bash
cd path/to/tau_amnlp/transformer_exercise
pip install torch sacrebleu sacremoses pandas tqdm
pip install --editable .
```

## Part 1: Training a Machine Translation Model

We will first train a baseline machine translation model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).
The IWSLT'14 dataset consists subtitles of TED talks translated to many languages. 

**Step 1:** Download and preprocess the data.
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
python fairseq_cli/preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```

**Step 2:** Train an encoder-decoder transformer using this data.
```bash
python train.py \
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
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --encoder-embed-dim 256 --decoder-embed-dim 256 \
    --encoder-layers 4 --decoder-layers 4 \
    --decoder-attention-heads 4 --encoder-attention-heads 4 \
    --no-epoch-checkpoints | tee baseline_train.log
```

* The specific configuration we will be using has 4 encoder/decoder layers and 4 attention heads for each multi-head attention sublayer. This amounts to 16 attention heads of each type (enc-enc, enc-dec, dec-dec), which are 48 heads overall. We use a smaller architecture then the default configuration to allow faster training.

* The ```--save-dir baseline``` argument will save the model into the "baseline" folder.

* If your GPU runs out of memory, you can reduce ```--max-tokens```.

**Step 3:** Evaluate the trained model.
```bash
python fairseq_cli/generate.py data-bin/iwslt14.tokenized.de-en \
--path baseline/checkpoint_best.pt --gen-subset valid \
--batch-size 128 --beam 5 --remove-bpe | tee baseline_gen.log
```

This step runs beam search and generates discrete strings. The system's outputs are compared to the reference strings in the test set using BLEU. You should be getting a BLEU score of about 33 to 34.

## Part 2: Masking Attention Heads

We would like to replicate one of the experiments in [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650), in which we mask a single attention head each time and test the model's performance.
To that end, we have added three command-line arguments to ```generate.py``` that specifies which attention head needs to be masked (see below). Your task is to mask the correct head given the argument.

Specifically, you should read the command line arguments (```args.mask_layer```, ```args.mask_head```, ```args.mask_layer_type```) in the TransformerModel's constructor and pass a masking flag to the relevant multi-head attention sublayer (only do so when the ```args.mask_layer_type``` equals to one of ```"enc-dec", "enc-enc" or "dec-dec"```  ). The transformer's implementation can be found in [fairseq/models/transformer.py](fairseq/models/transformer.py).

You will then need to implement the actual masking in [fairseq/modules/multihead_attention.py](fairseq/modules/multihead_attention.py), you might find this [blog post](http://jalammar.github.io/illustrated-transformer/) helpful.

The arguments ```mask_layer``` and ```mask_head``` specify the layer (0-3) and head (0-3) to mask, while ```mask_layer_type``` specifies which attention type is being masked ('enc-enc', 'enc-dec', 'dec-dec').

Run the following command to generate while masking the first head of the fourth decoder layer's enc-dec attention. You can also use this command without the `tee | baseline_mask.log` part to debug your code while working on it.
```bash
python fairseq_cli/generate.py data-bin/iwslt14.tokenized.de-en \
    --path baseline/checkpoint_best.pt --gen-subset valid \
    --batch-size 128 --beam 5 --remove-bpe \
    --model-overrides "{'mask_layer': 3, 'mask_head': 0, 'mask_layer_type': 'enc-dec'}" | tee baseline_mask.log
```
If your code works correctly, you should see a reduction in performance of about 1-3 BLEU.

When you are done, execute [check_all_masking_options.py](check_all_masking_options.py) as follows:
```bash
python check_all_masking_options.py data-bin/iwslt14.tokenized.de-en \
--path baseline/checkpoint_best.pt --gen-subset valid \
--batch-size 128 --beam 5 --remove-bpe --quiet | tee check_all_masking_options.log
```
This will run the same masking experiment for each attention head in the model, and print out a result table.

Take a look at the results and answer yourselves: What do we learn from these results? Which heads can be safely removed without incurring a significant change in performance (up to 0.5 BLEU)? Which heads are more critical? Is the trend consistent with the findings in the paper?

## Part 3: Reordering Transformer Sublayers

We would like to replicate the idea in [Improving Transformer Models by Reordering their Sublayers](https://arxiv.org/abs/1911.03864), where new transformer configurations can be created by chaining multi-head attention and feed-forward sublayers in arbitrary patterns. Your task is to implement this ability for the encoder in fairseq.

We have added the ```enc-layer-configuration```  argument to fairseq. It receives a string made of "A"s and "F"s, symbolizing multi-head attention feed-forward sublayers, respectively. For example, a regular encoder with 4 layers has the following architecture "AFAFAFAF".

Your implementation should be able to support any 8 (2 * number of encoder layers) letter sequence of "A"s and "F"s.

Train a sandwich transformer with the following command:

```bash
python train.py \
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
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --encoder-embed-dim 256 --decoder-embed-dim 256 \
    --encoder-layers 4 --decoder-layers 4 \
    --decoder-attention-heads 4 --encoder-attention-heads 4 \
    --no-epoch-checkpoints \
    --enc-layer-configuration AAAAAFFF | tee sandwich_train.log
```
Evaluate your new model with:

```bash
python fairseq_cli/generate.py data-bin/iwslt14.tokenized.de-en \
--path sandwich_baseline/checkpoint_best.pt --gen-subset valid \
--batch-size 128 --beam 5 --remove-bpe | tee sandwich_baseline_gen.log
```
Answer yourselves: Did the new architecture improve, hurt, or did not have a significant effect on performance? What about the number of parameters or the training speed? You are welcome to explore with different configurations, or implement (do not submit) this ability for the decoder also if you want.

You might want to test your implementation by training and evaluating a model with the default pattern of multi-head attention and feed-forward components and verify that you get the same results as those you got in part 1 (no need to submit this).

**Good luck!**