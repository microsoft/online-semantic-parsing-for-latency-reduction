# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch


def get_average_embeddings(final_layer, word2piece):

    # Average worpiece representations to get word representations
    num_words = len(word2piece)
    batch_dim, num_wordpieces, hidden_size = final_layer.shape
    assert batch_dim == 1, "batch_size must be 1"
    if num_words < num_wordpieces:
        word_features = torch.zeros(
            (1, num_words, hidden_size)
        ).to(final_layer.device)
        for word_idx, wordpiece_idx in enumerate(word2piece):
            # column of features for all involved worpieces
            column = final_layer[0:1, wordpiece_idx, :]
            if isinstance(wordpiece_idx, list):
                column = column.mean(1, keepdim=True)
            word_features[0:1, word_idx, :] = column
    else:
        word_features = final_layer

    return word_features


def get_wordpiece_to_word_map(sentence, roberta_bpe):

    # Get word and wordpiece tokens according to GPT2BPE (used by RoBERTa/BART)
    word_tokens = sentence.split()
    # NOTE this only returns the surface form of each byte encoding, which will not match as a subsequence of some
    #      characters such as '\x91' and chinese symbols -> we need to dynamically recover the chars from utf8 bytes
    # NOTE this is NOT used for matching
    # wordpiece_tokens = [
    #     roberta_bpe.decode(wordpiece)
    #     for wordpiece in roberta_bpe.encode(sentence).split()
    # ]
    # NOTE we need to use lower level bpe encodings to handle all characters such as chinese and u'\x91'
    #      the lower level bye bytes are used for matching
    wordpiece_bpe_ids = roberta_bpe.bpe.encode(sentence)    # List[int] corresponding to bpe vocab

    assert len(word_tokens) <= len(wordpiece_bpe_ids)
    assert isinstance(word_tokens, list)
    assert isinstance(wordpiece_bpe_ids, list)
    # assert isinstance(wordpiece_tokens, list)
    # assert len(wordpiece_tokens) == len(wordpiece_bpe_ids)

    w_index = 0
    word_to_wordpiece = []    # List[List[int]]
    subword_sequence = []
    bpe_id_sequence = []

    for wp_index, bpe_id in enumerate(wordpiece_bpe_ids):
        word = word_tokens[w_index]
        # only the initial word doesn't need whitespace at the beginning to be matched
        if w_index > 0:
            word = ' ' + word

        subword_sequence.append(wp_index)
        bpe_id_sequence.append(bpe_id)
        word_from_pieces = roberta_bpe.bpe.decode(bpe_id_sequence)    # this recovers any original characters
        if word == word_from_pieces:
            word_to_wordpiece.append(subword_sequence)
            w_index += 1
            subword_sequence = []
            bpe_id_sequence = []

    assert len(word_tokens) == len(word_to_wordpiece), 'word_to_wordpiece must be of the same size of the word_tokens'
    assert word_to_wordpiece[0][0] == 0 and word_to_wordpiece[-1][-1] == len(wordpiece_bpe_ids) - 1, \
        'word_to_wordpiece mapping must cover all wordpieces, from the beginning towards the end'

    return word_to_wordpiece


def get_scatter_indices(word2piece, reverse=False):
    if reverse:
        indices = range(len(word2piece))[::-1]
    else:
        indices = range(len(word2piece))
    # we will need as well the wordpiece to word indices
    wp_indices = [
        [index] * (len(span) if isinstance(span, list) else 1)
        for index, span in zip(indices, word2piece)
    ]
    # flatten the list
    wp_indices = [x for span in wp_indices for x in span]
    return torch.tensor(wp_indices)


class SentenceEncodingBART:
    def __init__(self, name):
        # bart model name
        self.name = name

        if name in ['bart.base', 'bart.large']:
            self.model = torch.hub.load('pytorch/fairseq', name)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.cuda()
                print(f'Using {name} extraction in GPU')
            else:
                print(f'Using {name} extraction in cpu (slow, wont OOM)')
        else:
            raise Exception(f'Unknown pretrained model name or path {name}')

    def encode_sentence(self, sentence_string):
        """BPE tokenization and numerical encoding based on model vocabulary.
        Args:
            sentence_string (str): sentence string, not tokenized.
        Raises:
            Exception: [description]
            NotImplementedError: [description]
        Returns:
            [type]: [description]
        """
        # get numerical encoding of the sentence
        # NOTE bpe token ids include BOS `<s>` and EOS `</s>`
        wordpiece_ids = self.model.encode(sentence_string)

        # get word to word piece mapping
        # NOTE the mapping index does not consider BOS `<s>` and EOS `</s>`
        word2piece = get_wordpiece_to_word_map(
            sentence_string,
            self.model.bpe
        )

        return wordpiece_ids, word2piece
