# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    # TransformerDecoderLayer,
    # TransformerEncoderLayer,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .transformer_tgt_pointer import TransformerTgtPointerModel, TransformerDecoder, transformer_pointer
from fairseq_gap.constants import SpecialSymbols


@register_model("transformer_tgt_pointer_scp")
class TransformerTgtPointerSrcCopyModel(TransformerTgtPointerModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerTgtPointerModel.add_args(parser)

        # additional: tgt src copy distribution from decoder cross attention
        parser.add_argument('--tgt-src-copy-layers', nargs='*', type=int,
                            help='target source copy in decoder cross-attention: which layers to use')
        parser.add_argument('--tgt-src-copy-heads', type=int,
                            help='target source copy in decoder cross-attention: how many heads per layer to use')

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerSrcCopyDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        # customized
        src_fix_emb: Optional[torch.Tensor] = None,
        src_wordpieces: torch.Tensor = None,
        src_wp2w: torch.Tensor = None,
        tgt_vocab_masks: torch.Tensor = None,
        tgt_actnode_masks: torch.Tensor = None,
        tgt_src_cursors: torch.Tensor = None,
        tgt_src_pos: torch.Tensor = None,
        # unused
        **unused
    ):
        """
        Run the forward pass for an encoder-decoder model.
        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens,
            # customized
            src_fix_emb=src_fix_emb,
            src_wordpieces=src_wordpieces,
            src_wp2w=src_wp2w
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            # customized
            tgt_vocab_masks=tgt_vocab_masks,
            tgt_actnode_masks=tgt_actnode_masks,
            tgt_src_cursors=tgt_src_cursors,
            tgt_src_pos=tgt_src_pos,
        )
        return decoder_out


class TransformerSrcCopyDecoder(TransformerDecoder):

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        # customized: specific to GAP (Graph/Generalized Action-Pointer)
        tgt_vocab_masks=None,
        tgt_actnode_masks=None,
        tgt_src_cursors=None,
        tgt_src_pos=None,
        # unused
        **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            # customized for GAP
            tgt_actnode_masks=tgt_actnode_masks,
            tgt_src_cursors=tgt_src_cursors,
        )

        if not features_only:
            x = self.output_layer(
                x,
                # customized for GAP
                tgt_vocab_masks=tgt_vocab_masks,
                # for src copy distribution
                extra=extra,
                )
        return x, extra

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        # customized for GAP
        tgt_actnode_masks=None,
        tgt_src_cursors=None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        # ========== combine the corresponding source token embeddings with the action embeddings as input ==========
        if self.args.apply_tgt_input_src:
            assert self.args.tgt_input_src_emb == 'top' and self.args.tgt_input_src_combine == 'add', \
                'currently we do not support other variations (which may have a bit of extra parameters'

            # 1) take out the source embeddings
            src_embs = encoder_out.encoder_out.transpose(0, 1)    # size (batch_size, src_max_len, encoder_emb_dim)
            if not self.args.tgt_input_src_backprop:
                src_embs = src_embs.detach()

            # 2) align the source embeddings to the tgt input actions
            assert tgt_src_cursors is not None
            tgt_src_index = tgt_src_cursors.clone()    # size (bsz, tgt_max_len)
            if encoder_out.encoder_padding_mask is not None:
                src_num_pads = encoder_out.encoder_padding_mask.sum(dim=1, keepdim=True)
                tgt_src_index = tgt_src_index + src_num_pads    # NOTE this is key to left padding!

            # NOTE due to padding value is 1, the indexes could be out of range of src_max_len ->
            #      we fix invalid indexes for padding positions (invalid should only happen at padding positions,
            #      and when the src sentence has max length 1)
            tgt_src_index[tgt_src_index >= src_embs.size(1)] = src_embs.size(1) - 1

            tgt_src_index = tgt_src_index.unsqueeze(-1).repeat(1, 1, src_embs.size(-1))
            # or
            # tgt_src_index = tgt_src_index.unsqueeze(-1).expand(-1, -1, src_embs.size(-1))

            # # NOTE deal with the corner case when the max_src_len in the whole batch is only 1 ->
            # #      already dealt with above!
            # if encoder_out.encoder_out.size(0) == 1:
            #     # NOTE we have to fix all indexes at 0 (including the padding positions)!!
            #     #      (the default padding value is 1, which would cause an index out of range error hard to debug)
            #     tgt_src_index.fill_(0)

            src_embs = torch.gather(src_embs, 1, tgt_src_index)
            # size (bsz, tgt_max_len, src_embs.size(-1))

            # 3) combine the action embeddings with the aligned source token embeddings
            if self.args.tgt_input_src_combine == 'cat':
                x = self.combine_src_embs(torch.cat([src_embs, x], dim=-1))    # NOTE not initialized
            elif self.args.tgt_input_src_combine == 'add':
                x = src_embs + x
            else:
                raise NotImplementedError
        # ===========================================================================================================

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # ========== alignment guidance in the cross-attention: get the mask ==========
        if self.args.apply_tgt_src_align:
            assert tgt_src_cursors is not None
            cross_attention_mask = get_cross_attention_mask_heads(tgt_src_cursors,
                                                                  encoder_out.encoder_out.size(0),
                                                                  encoder_out.encoder_padding_mask,
                                                                  self.args.tgt_src_align_focus,
                                                                  self.args.tgt_src_align_heads,
                                                                  self.layers[0].encoder_attn.num_heads)
        else:
            cross_attention_mask = None
        # ==============================================================================

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]

        # for pointer distribution
        attn_ptr = None
        attn_all_ptr = []

        # for cross attention copying distribution
        attn_src = None
        attn_src_all = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # change the decoder layer to output both cross_attention (as in default case)
            # and the decoder self attention
            x, layer_attn, _, self_attn = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                # need_attn=bool((idx == alignment_layer)),
                need_attn=True,    # to return src attention "layer_attn"
                need_head_weights=bool((idx == alignment_layer)),
                # customized
                cross_attention_mask=(cross_attention_mask
                                      if idx in self.args.tgt_src_align_layers
                                      else None),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            # ========== for pointer distribution ==========
            if idx in self.args.pointer_dist_decoder_selfattn_layers:

                # attn is tgt self-attention of size (bsz, num_heads, tgt_len, tgt_len) with future masks
                if self.args.pointer_dist_decoder_selfattn_heads == 1:
                    attn_ptr = self_attn[:, 0, :, :]
                    attn_all_ptr.append(attn_ptr)
                else:
                    attn_ptr = self_attn[:, :self.args.pointer_dist_decoder_selfattn_heads, :, :]
                    if self.args.pointer_dist_decoder_selfattn_avg == 1:
                        # arithmetic mean
                        attn_ptr = attn_ptr.sum(dim=1) / self.args.pointer_dist_decoder_selfattn_heads
                        attn_all_ptr.append(attn_ptr)
                    elif self.args.pointer_dist_decoder_selfattn_avg == 0:
                        # geometric mean
                        attn_ptr = attn_ptr.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                        # TODO there is an nan bug when backward for the above power
                        attn_all_ptr.append(attn_ptr)
                    elif self.args.pointer_dist_decoder_selfattn_avg == -1:
                        # no mean
                        pointer_dists = list(map(
                            lambda x: x.squeeze(1),
                            torch.chunk(attn_ptr, self.args.pointer_dist_decoder_selfattn_heads, dim=1)))
                        # for decoding: using a single pointer distribution
                        attn_ptr = attn_ptr.prod(dim=1).pow(1 / self.args.pointer_dist_decoder_selfattn_heads)
                        attn_all_ptr.extend(pointer_dists)
                    else:
                        raise ValueError

            # breakpoint()
            # ========== for src pointer distribution from cross attention distribution ==========
            if idx in self.args.tgt_src_copy_layers:

                # attn is cross-attention of size (bsz, num_heads, tgt_len, src_len)
                if self.args.tgt_src_copy_heads == 1:
                    attn_src = None
                    attn_src_all.append(layer_attn[:, 0, :, :])
                else:
                    attn_src = layer_attn[:, :self.args.tgt_src_copy_heads, :, :]
                    # no mean
                    align_dists = list(map(
                        lambda x: x.squeeze(1),
                        torch.chunk(attn_src, self.args.tgt_src_copy_heads, dim=1)))
                    attn_src_all.extend(align_dists)

        # for decoding: which pointer distribution to use
        attn_ptr = attn_all_ptr[self.args.pointer_dist_decoder_selfattn_layers.index(
            self.args.pointer_dist_decoder_selfattn_infer)]

        # for decoding: which copying distribution to use
        # TODO currently we fix at last layer: add a flag
        # assert self.args.tgt_src_copy_heads == 1
        # assert len(self.args.tgt_src_copy_layers) == 1
        # attn_src = attn_src_all[-1]    # NOTE do not use this -> it would only feed one distribution into loss
        # average
        attn_src = sum(attn_src_all) / len(attn_src_all)

        # ====================================================

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        # NOTE here 'attn_ptr' is used for inference pointer prediction, 'attn_all_ptr' is used for loss calculation
        # TODO change the names to be more straightforward, such as 'pointer_dist_infer', 'pointer_dist_list'
        # TODO add teacher forcing; this will change the backward behavior
        # change the original output TODO change the names to include both original `attn` and that for pointer
        # return x, {"attn": [attn], "inner_states": inner_states}
        return x, {'attn': attn_ptr, 'inner_states': inner_states, 'attn_all': attn_all_ptr,
                   'attn_src': attn_src, 'attn_src_all': attn_src_all}

    def output_layer(self, features, tgt_vocab_masks=None, extra=None):
        """Project features to the vocabulary size.

        Deal with source copy distribution.
        """
        if self.adaptive_softmax is None:
            # project back to size of vocabulary, of size (bsz, tgt_max_len, vocab_size)
            out = self.output_projection(features)

            if self.args.apply_tgt_vocab_masks:
                assert tgt_vocab_masks is not None
                out[tgt_vocab_masks == 0] = float('-inf')

            # take out the src copy distributions
            assert extra is not None
            # normalized probability scores, of size (bsz, tgt_max_len, src_max_len)
            scp_dist = extra['attn_src']

            # NOTE this is for numerical stability; otherwise log backward will get nan (crucial !!!)
            scp_dist = scp_dist.clamp(min=1e-8)

            # transform everything to log space
            lprobs = self.get_normalized_probs(net_output=(out,), log_probs=True)
            lprobs_scp = torch.log(scp_dist)

            # take out COPY action probability, combine with src pointer distribution
            copy_index = self.dictionary.indices[SpecialSymbols.COPY]
            lprobs_scp = lprobs[:, :, copy_index].unsqueeze(2) + lprobs_scp

            # set the COPY action probability to 0 (log to -inf)
            # below caused an error: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # (it should be fine not to set the prob of bare COPY action to -inf, as it will never be the target.
            #  However, label smoothing refers to '-inf' cases)
            # lprobs[:, :, copy_index] = float('-inf')

            # maskout_copy = torch.ones_like(lprobs)
            # maskout_copy[:, :, copy_index] = float('-inf')
            # lprobs = lprobs * maskout_copy    # This would cause +inf, since some copies are already masked out...

            maskout_copy = torch.zeros_like(lprobs)
            maskout_copy[:, :, copy_index] = 1
            maskout_copy[maskout_copy == 1] = float('-inf')

            # concatenate the vocabulary distribution with src copy distribution
            lprobs_cat = torch.cat([lprobs, lprobs_scp], dim=2)

            return lprobs_cat
        else:
            assert not self.args.apply_tgt_vocab_masks
            return features


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp")
def transformer_pointer_scp(args):
    # basic arguments
    transformer_pointer(args)

    # additional: tgt src copy distribution from decoder cross attention
    # default to penultimate layer
    args.tgt_src_copy_layers = getattr(args, 'tgt_src_copy_layers', args.decoder_layers - 2)
    args.tgt_src_copy_heads = getattr(args, 'tgt_src_copy_heads', 1)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_6x6x4hx128x512")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_6x6x8hx512x2048")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_3x3x4hx256x512")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_3x3x4hx128x512")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 3)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_2x2x4hx256x512")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)


@register_model_architecture("transformer_tgt_pointer_scp", "transformer_tgt_pointer_scp_2x2x4hx128x512")
def transformer_pointer_scp_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    transformer_pointer_scp(args)
