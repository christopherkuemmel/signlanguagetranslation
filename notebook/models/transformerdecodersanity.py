"""Sanity check the transformer decoder."""

import torch

import data.transforms as tr
from models.sign2text import TranslationDecoder
from rosita import binarize, debinarize

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seed for reproducability
torch.manual_seed(42)

SENTENCES = [
    'Wikidata is a free and open knowledge base that can be read and edited by both humans and machines.',
    'Wikidata acts as central storage for the structured data of its Wikimedia sister projects including Wikipedia, Wikivoyage, Wiktionary, Wikisource, and others.',
    'Wikidata also provides support to many other sites and services beyond just Wikimedia projects!',
    'The content of Wikidata is available under a free license, exported using standard formats, and can be interlinked to other open data sets on the linked data web.',
]


def compare_encode():
    """Compare encoding steps (tokenization, bpe, ..) from fairseq to our own implementation."""
    fairseq_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')

    encoder = TranslationDecoder()
    tokenize = tr.Tokenize('de')
    apply_bpe = tr.ApplyBPE('data/bpe/bpecodes', 'data/bpe/dict.de.txt')

    for sentence in SENTENCES:
        # create a rosita like dict
        rosita_out = {'translation': sentence}

        rosita_out = tokenize(rosita_out)
        rosita_out = apply_bpe(rosita_out)

        rosita_out['translation'] = [(word, ) for word in rosita_out['translation']]
        # expand batch dim for rosita dict, and only give translations
        # then remove batch dim
        rosita_out = binarize([rosita_out['translation']], encoder.encoder.dictionary, add_bos=False).squeeze()

        ground_truth = fairseq_model.encode(sentence)

        print(f"Rosita encode: {rosita_out}")
        print(f"Fairseq encode: {ground_truth}")
        assert (rosita_out == ground_truth).all()


def compare_decode():
    """Compare decoding steps (detokenization, remove bpe, ..) from fairseq to our own implementation."""
    fairseq_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')

    encoder = TranslationDecoder()
    detokenize = tr.Detokenize('de')
    remove_bpe = tr.RemoveBPE()

    for sentence in ENCODED_SENTENCES:
        rosita_out = debinarize([sentence], encoder.encoder.dictionary)
        # remove tuples
        rosita_out = [[token[0] for token in sentence] for sentence in rosita_out][0]

        # create a rosita like dict
        rosita_out = {'translation': rosita_out}
        rosita_out = remove_bpe(rosita_out)
        rosita_out = ' '.join(detokenize(rosita_out)['translation'])

        ground_truth = fairseq_model.decode(sentence)
        print(f"Rosita encode: {rosita_out}")
        print(f"Fairseq encode: {ground_truth}")
        assert rosita_out == ground_truth


def check_transformer_decoder():
    """Check ResNext 101 3d kinetics."""

    with torch.no_grad():
        encoder = TranslationDecoder()


if __name__ == "__main__":
    compare_encode()
    compare_decode()
    check_transformer_decoder()

### Results

## Notes
