import numpy as np
from examples.noisychannel.rerank_utils import parse_bleu_scoring

from fairseq import options
from fairseq_cli.generate import main
import pandas as pd


def mask_all_heads_combination():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    args.quiet = True
    number_of_transformer_layers = 4
    number_of_attention_heads = 4
    mask_layer_combinations = ['enc-enc', 'enc-dec', 'dec-dec']
    results_dict = {i: np.zeros((number_of_transformer_layers, number_of_attention_heads)) for i in
                    mask_layer_combinations}
    for i in mask_layer_combinations:
        for j in range(number_of_transformer_layers):
            for k in range(number_of_attention_heads):
                args.model_overrides = str({"mask_layer": j, "mask_head": k, "mask_layer_type": i})
                scorer = main(args)
                results_dict[i][j][k] = float(parse_bleu_scoring(scorer.result_string()))

    for name in mask_layer_combinations:
        print("table of score with masking {} attention head".format(name))
        print("rows are transformer layer number and columns are head number".format(name))
        df = pd.DataFrame(data=results_dict[name], index=[str(j) for j in range(number_of_transformer_layers)],
                          columns=[str(k) for k in range(number_of_attention_heads)])
        print(df)


if __name__ == '__main__':
    mask_all_heads_combination()

