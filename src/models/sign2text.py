"""Contains the model architecture for the sign2text task."""

import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnext_101_kinetics import load_resnext101_3d_kinetics

# NamedTuple implementation taken from https://github.com/pytorch/fairseq/blob/master/fairseq/models/fairseq_encoder.py#L10
EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", torch.Tensor),  # T x B x C
        ("encoder_padding_mask", torch.Tensor),  # B x T
        ("encoder_embedding", torch.Tensor),  # B x T x C
        ("encoder_states", Optional[List[torch.Tensor]]),  # List[T x B x C]
    ],
)


class FeatureExtractor(nn.Module):
    """3D ResNext FeatureExtractor pre-trainend on kinetics dataset."""
    def __init__(self, image_size: int, window_size: int, output_features: int) -> None:
        """
        Args:
            image_size (int): The dimension of the images (height and width - needs to be square).
            window_size (int): The dimension of the sliding depth window.
            output_features (int): The number of output features.
            # TODO: cut_off_layers (int): Which layer to trim in the ResNext model
        """
        super(FeatureExtractor, self).__init__()

        # TODO: trim/ cut_off_layer
        # TODO: adaptive avgpool
        self.feature_extractor = load_resnext101_3d_kinetics(image_size, window_size, 1)
        self.linear = nn.Linear(2048, output_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ResNext101_3D forward step.

        Args:
            images (torch.Tensor): The input images with the size (seq_len, color, depth, height, width).
                Note that the resnext does not support a batch dim. Therefore, you need to adjust the input tensor upfront.

        Returns:
            torch.Tensor: The output from the ResNext101 3D with the size (seq_len, output_features)
        """
        images = self.feature_extractor(images)
        return self.linear(images.squeeze())


class Identity(nn.Module):
    """Identity helper module which simply returns the given input"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, sample) -> torch.Tensor:
        """Identity forward step.

        Args:
            sample (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor.
        """
        return sample


class PositionalEmbedding(nn.Module):
    """Positional Embedding Module implementation.

    This is a modified and slightly extended version of
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, embedding_dim: int = 1024, max_len: int = 100):
        super(PositionalEmbedding, self).__init__()

        positional_embedding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        positional_embedding = positional_embedding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('positional_embedding', positional_embedding)

    def forward(self, sample: torch.Tensor):
        """Positional Encoder forward step.
        Args:
            sample (torch.Tensor): The input sample with size (source_length, batch_size, embedding_dim)

        Returns:
            torch.Tensor: The fixed positional embeddings with size (source_length, batch_size = 1, embedding_dim).
                If you apply this tensor on others it need to be broadcasted on the batch dimension. Make sure they match.
        """
        return self.positional_embedding[:sample.size(0), :]


class TranslationEncoder(nn.Module):
    """Translation Transformer Encoder pre-trained for english-german translation.

    This is a modified and slightly extended version of
    https://github.com/pytorch/fairseq/blob/f1d856e006a0fcb2b22afaad2eb456a671204557/fairseq/models/transformer.py#L311

    Using pre-trained models from
    https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
    """
    def __init__(self, translation_model: str = 'transformer.wmt19.en-de.single_model', embedding_dim: int = 1024, from_scratch: bool = False) -> None:
        """
        Args:
            translation_model (str): The pre-trained translation architecture.
                Only the encoder part will be taken for this model!
                Default is `transformer.wmt19.en-de.single_model`.
            embedding_dim (int): The dimension of the input embeddings.
                Only works if model when `from_scratch` equals True.
                Otherwise the dimension is taken from the pre-trained model.
            from_scratch (bool): Flush pre-trained weights and train from_scratch.
        """
        super(TranslationEncoder, self).__init__()

        # TODO: input params
        self.dropout = 0.2
        self.encoder_layerdrop = 0

        # TODO: refactor to not start based on loaded architecture
        # better define model and then load the params
        self.encoder = torch.hub.load('pytorch/fairseq', translation_model).models[0].encoder

        # TODO: flush pre-trained encoder weights
        #? better apply from outer scope?
        if from_scratch:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = self.encoder.layers[0].embed_dim

        # remove input embeddings and replace with identity model
        self.encoder.embed_tokens = Identity()
        # set positional embedding for
        self.encoder.embed_positions = PositionalEmbedding(self.embedding_dim)

    def forward(self, sample: torch.Tensor, padding_mask: torch.BoolTensor, return_all_hiddens: Optional[bool] = False) -> NamedTuple:
        """TranslationEncoder forward step.
        Args:
            sample (torch.Tensor): The input embeddings with size (source_length, batch_size, embedding_dim)
                `source_length` == sentence_length || sequence_length
            padding_mask (torch.BoolTensor): The padding mask for the input embeddings with size (batch_size, sequence_length).
                Where 1 = value was padded and therefore, will be ignored.
            return_all_hiddens (Optional[bool]): Also return all of the intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): The last encoder layer's output of shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): The positions of padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): The (scaled) embedding lookup of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): All intermediate hidden states of shape `(src_len, batch, embed_dim)`.
                    Only populated if *return_all_hiddens* is True.
        """
        # positional embeddings
        #! sample = embedding = self.encoder.embed_scale * self.encoder.embed_tokens(sample)
        embedding = self.encoder.embed_scale * self.encoder.embed_tokens(sample)
        if self.encoder.embed_positions is not None:
            sample = embedding + self.encoder.embed_positions(sample)
        sample = F.dropout(sample, p=self.dropout, training=self.training)

        encoder_states = [] if return_all_hiddens else None

        #!
        sample = sample.transpose(0, 1)

        # encoder layers
        for layer in self.encoder.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                # sample -> T x B x C && padding_mask -> B x T
                sample = layer(sample, padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(sample)

        return EncoderOut(
            encoder_out=sample,  # T x B x C
            encoder_padding_mask=padding_mask,  # B x T
            encoder_embedding=embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )


class TranslationDecoder(nn.Module):
    """Translation Transformer Decoder pre-trained for english-german translation.

    Using pre-trained models from
    https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md

    This is a modified and extended version of
    https://github.com/pytorch/fairseq/blob/f1d856e006a0fcb2b22afaad2eb456a671204557/fairseq/models/transformer.py#L490
    """
    def __init__(self, translation_model: str = 'transformer.wmt19.en-de.single_model', from_scratch: bool = False) -> None:
        """
        Args:
            translation_model (str): The pre-trained translation architecture.
                Only the decoder part will be taken for this model!
                Default is `transformer.wmt19.en-de.single_model`.
            from_scratch (bool): Flush pre-trained weights and train from_scratch.
        """
        super(TranslationDecoder, self).__init__()

        self.decoder = torch.hub.load('pytorch/fairseq', translation_model).models[0].decoder
        self.vocab = self.decoder.dictionary

        # TODO: flush pre-trained encoder weights
        # if from_scratch:
        #     model.apply(init_weights)
        #     pass

    def forward(self,
                previous_tokens: torch.LongTensor,
                sample: EncoderOut,
                incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
                return_all_hiddens: bool = False) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """TranslationDecoder forward step.
        Args:
            previous_tokens (torch.LongTensor): The previous decoder outputs (tokens) as LongTensor with the size (batch, target_lenth)
                `target_length` specifies how many outputs should be generated.
                If `target_length` > 1 teacher forcing will be applied!
            sample (EncoderOut): NamedTuple of encoder outputs.
            incremental_state (Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]):
                The dictionary of the last decoder state (for incremental decoding steps).
            return_all_hiddens (bool): Also return all of the intermediate hidden states (default: False).

        Returns:
            Tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder(previous_tokens, sample, incremental_state, return_all_hiddens=return_all_hiddens)


class Sign2Text(nn.Module):
    """Sign2Text model implementation."""
    def __init__(self, feature_extractor: nn.Module, encoder: nn.Module, decoder: nn.Module):
        """
        Args:
            feature_extractor (nn.Module): The feature extractor module.
            encoder (nn.Module): The TransformerEncoder module.
            decoder (nn.Module): The TransformerDecoder module.
        """
        super(Sign2Text, self).__init__()

        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            sample: torch.Tensor,
            sample_mask: torch.BoolTensor,
            previous_tokens: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """Sign2Text forward step.

        Args:
            sample (torch.Tensor): The input video with size (batch_size, sequence_length, color, window_depth, heigth, width).
            sample_mask (torch.BoolTensor): The padding mask for the input embeddings with size (batch_size, sequence_length).
                Were 1 = value was padded and therefore, will be ignored.
            previous_tokens (torch.Tensor): The previous decoder outputs (tokens) as LongTensor with the size (batch, target_lenth).
                `target_length` specifies how many outputs should be generated.
                If `target_length` > 1 teacher forcing will be applied!

        Returns:
            Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
                - The predicted output tokens as LongTensor with size (batch_size, target_length, vocab_dim).
                - A dictionary with keys `attn` and `inner_states`.
                    Where the value of `attn` is a torch.Tensor,
                    And the value of `inner_states` is a List of torch.Tensor.
        """
        batch_size, sequence_length, color, window_depth, heigth, width = sample.size()

        # flatten batch and seq_len
        sample = sample.view(-1, color, window_depth, heigth, width)

        # extract features
        sample = self.feature_extractor(sample)

        # undo flatten and permute B x T x E -> T x B x E
        sample = sample.view(batch_size, sequence_length, -1).permute(1, 0, 2)

        # encoder forward -> EncoderOut
        sample = self.encoder(sample, sample_mask, True)

        # decoder forward
        # decoder returns tuple:
        #   - the decoder's output of shape `(batch, tgt_len, vocab)`
        #   - a dictionary with any model-specific outputs
        return self.decoder(previous_tokens, sample)
