"""Sanity check the transformer encoder."""

import torch

import data.transforms as tr
from models.sign2text import TranslationEncoder
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
ENCODED_SENTENCES = [
    [18304, 798, 22, 19, 510, 11, 1429, 3412, 4065, 38, 109, 44, 3223, 11, 2698, 764, 47, 1026, 15631, 11, 5623, 5, 2],
    [
        18304, 798, 8210, 41, 3543, 4558, 23, 6, 3854, 7592, 798, 9, 187, 14751, 559, 2698, 19, 14778, 2358, 1290, 18304, 18227, 4, 18304, 714, 167, 446, 4,
        14751, 9137, 4, 14751, 590, 8013, 774, 4, 11, 2785, 5, 2
    ],
    [18304, 798, 157, 2628, 897, 16, 545, 288, 2823, 11, 1001, 6438, 685, 14751, 559, 2698, 19, 2358, 111, 2],
    [
        31, 2606, 9, 18304, 798, 22, 652, 757, 19, 510, 12880, 4, 13156, 764, 1332, 2345, 646, 966, 94, 4, 11, 109, 44, 476, 9105, 16, 288, 1429, 798, 6660, 26,
        6, 9105, 798, 3834, 5, 2
    ],
    [
        31, 2606, 9, 18304, 798, 22, 652, 757, 19, 510, 12880, 4, 13156, 764, 1332, 2345, 646, 966, 94, 4, 11, 109, 44, 476, 9105, 16, 288, 1429, 798, 6660, 26,
        6, 9105, 798, 3834, 5, 2
    ],
]


def compare_encode():
    """Compare encoding steps (tokenization, bpe, ..) from fairseq to our own implementation."""
    fairseq_model = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')

    encoder = TranslationEncoder()
    tokenize = tr.Tokenize('en')
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

    encoder = TranslationEncoder()
    detokenize = tr.Detokenize('en')
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


def check_transformer_encoder():
    """Check ResNext 101 3d kinetics."""

    with torch.no_grad():
        encoder = TranslationEncoder()

        # we cut of the embedding layer + positional embedding in our own impl
        # therefore we need to "reapply" those
        # the forward path should be the same as in our own impl
        fairseq_encoder = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model').models[0].encoder
        encoder.encoder.embed_tokens = fairseq_encoder.embed_tokens
        encoder.encoder.embed_positions = fairseq_encoder.embed_positions

        encoder.eval()

        tokenize = tr.Tokenize('en')
        apply_bpe = tr.ApplyBPE('data/bpe/bpecodes', 'data/bpe/dict.de.txt')

        for sentence in SENTENCES:
            # create a rosita like dict
            rosita_out = {'translation': sentence}

            rosita_out = tokenize(rosita_out)
            rosita_out = apply_bpe(rosita_out)

            rosita_out['translation'] = [(word, ) for word in rosita_out['translation']]
            # expand batch dim for rosita dict, and only give translations
            # then remove batch dim
            rosita_out = binarize([rosita_out['translation']], encoder.encoder.dictionary, add_bos=False)
            rosita_out_mask = torch.zeros(rosita_out.size()).bool()

            ground_truth = fairseq_encoder(rosita_out, rosita_out_mask)
            rosita_out = encoder(rosita_out, rosita_out_mask)

            assert (ground_truth.encoder_embedding == rosita_out.encoder_embedding).all()
            assert (ground_truth.encoder_out == rosita_out.encoder_out).all()

            #? evaluate performance of embeddings?
            # We could combine token-level embeddings to sentence level and evaluate this with SentEval
            # https://github.com/facebookresearch/SentEval


if __name__ == "__main__":
    compare_encode()
    compare_decode()
    check_transformer_encoder()

### Results

## Notes
# * do we need to add bos token when encoding? -> probably not
