"""Pytorch Transformer Playground."""
import torch

## BERT ###
from transformers import BertTokenizer, BertModel, BertForPreTraining
from transformers import PreTrainedEncoderDecoder


def load_bert():
    """Loads huggingfaces' bert transformer"""
    # Load pretrained model/tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
    model = BertModel.from_pretrained('bert-base-german-cased')
    # model = BertModel.from_pretrained('bert-base-german-cased', is_decoder=True)
    # model = BertForPreTraining.from_pretrained('bert-base-german-cased')

    # Encode text
    # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
    input_ids = torch.tensor([tokenizer.encode("Hier ist ein Text zum encodieren.", add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples


def load_fairseq():
    """Loads fairseq translation and language models."""
    ### FAIRSEQ ###
    # load pre-trained german language model
    # https://github.com/pytorch/fairseq/tree/master/examples/wmt19
    de_lm = torch.hub.load('pytorch/fairseq', 'transformer_lm.wmt19.de', tokenizer='moses', bpe='fastbpe')
    de_lm.sample("Maschinelles lernen ist")

    # load pre-trained english-german translation model
    en_de_lm = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
    en_de_lm.sample("Hello World, this is a longer test sequence.")


if __name__ == "__main__":
    load_bert()
    load_fairseq()
