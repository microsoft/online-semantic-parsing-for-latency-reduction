# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch

from .sentence_encoding_bart import get_wordpiece_to_word_map, get_average_embeddings, get_scatter_indices
from ..utils_font import yellow_font


class SentenceEmbeddingRoberta:

    def __init__(self, name, bert_layers, model=None, remove_be=False, avg_word=False):

        # embedding type name
        self.name = name
        # select some layers for averaging
        self.bert_layers = bert_layers

        if model is None:
            if name in ['roberta.base', 'roberta.large']:

                # Extract
                self.roberta = torch.hub.load('pytorch/fairseq', name)
                self.roberta.eval()
                if torch.cuda.is_available():
                    self.roberta.cuda()
                    print(f'Using {name} extraction in GPU')
                else:
                    print(f'Using {name} extraction in cpu (slow, wont OOM)')

            else:
                raise Exception(
                    f'Unknown --pretrained-embed {name}'
                )
        else:
            self.roberta = model

        self.remove_be = remove_be    # whether to remove <s> and </s>
        self.avg_word = avg_word      # whether to average the embeddings from subtokens to words

    def extract_features(self, wordpieces):
        """Extract features from wordpieces"""

        if self.bert_layers is None:
            # normal RoBERTa
            return self.roberta.extract_features(wordpieces)
        else:
            # layer average RoBERTa
            features = self.roberta.extract_features(
                wordpieces,
                return_all_hiddens=True
            )
            # sum layers
            feature_layers = []
            for layer_index in self.bert_layers:
                feature_layers.append(features[layer_index])
            feature_layers = sum(feature_layers)
            return torch.div(feature_layers, len(self.bert_layers))

    def extract(self, sentence_string):
        """
        sentence_string (not tokenized)
        """

        # get words, wordpieces and mapping
        # FIXME: PTB oracle already tokenized
        word2piece = get_wordpiece_to_word_map(
            sentence_string,
            self.roberta.bpe
        )

        # NOTE: We need to re-extract BPE inside roberta. Token indices
        # will also be different. BOS/EOS added
        wordpieces_roberta = self.roberta.encode(sentence_string)

        # Extract roberta, remove BOS/EOS
        if torch.cuda.is_available():

            # Hotfix for sequences above 512
            if wordpieces_roberta.shape[0] > 512:
                excess = wordpieces_roberta.shape[0] - 512
                # first 512 tokens
                last_layer = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)[:512]
                )
                # last 512 tokens
                last_layer2 = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)[excess:]
                )
                # concatenate
                shape = (last_layer, last_layer2[:, -excess:, :])
                last_layer = torch.cat(shape, 1)

                assert wordpieces_roberta.shape[0] == last_layer.shape[1]

                # warn user about this
                string = '\nMAX_POS overflow!! {wordpieces_roberta.shape[0]}'
                print(yellow_font(string))

            else:

                # Normal extraction
                last_layer = self.extract_features(
                    wordpieces_roberta.to(self.roberta.device)
                )

        else:

            # Copy code above
            raise NotImplementedError()
            last_layer = self.roberta.extract_features(
                wordpieces_roberta
            )

        # FIXME: this should not bee needed using roberta.eval()
        last_layer = last_layer.detach()

        # Ignore start and end symbols
        if self.remove_be:
            last_layer = last_layer[0:1, 1:-1, :]

        # average over wordpieces of same word
        if self.avg_word:
            assert self.remove_be
            word_features = get_average_embeddings(
                last_layer,
                word2piece
            )
        else:
            word_features = last_layer

    #    # sanity check differentiable and non differentiable averaging
    #    match
    #    from torch_scatter import scatter_mean
    #    word_features2 = scatter_mean(
    #        last_layer[0, :, :],
    #        get_scatter_indices(word2piece).to(roberta.device),
    #        dim=0
    #    )
    #    # This works
    #    assert np.allclose(word_features.cpu(), word_features2.cpu())

        return word_features, wordpieces_roberta, word2piece
