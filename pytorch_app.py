# work out if still want to sanic and other libaies for you version

# TODO: Edit evaluate function to allow for saved models DONE
# TODO: import picke objects: word2count, word2index, input_lang, output_lang DONE
# TODO: Set plotting attention matrices


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

# Comment pytorch moducles while testing on local machine.

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
# import index and count
MAX_LENGTH = 10
# word2index = {}
# word2count = {}

# import pickled objects
# https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python/6568495


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


LANG_PATH = 'Input_outputs_langs/'
# Input_outputs_langs/input_lang.pkl


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'Lang':
            from pytorch_app import Lang
            return Lang
        return super().find_class(module, name)


try:
    infile_input_lang = open('Input_outputs_langs/input_lang.pkl', 'rb')
    input_lang = pickle.load(infile_input_lang)
    infile_input_lang.close()
except AttributeError:
    print('Using CustomUnpickler')
    input_lang = CustomUnpickler(open('Input_outputs_langs/input_lang.pkl', 'rb')).load()

infile_output_lang = open('Input_outputs_langs/output_lang.pkl', 'rb')
output_lang = pickle.load(infile_output_lang)
infile_output_lang.close()

print('--------\nTesting pickle loading: ')
print('input lang', input_lang)
print('output lang', output_lang)

# input_lang = None
# output_lang = None


# https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()

# print('-------------\n DEBUGING:')
# print('input_lang attrubites:\n')
# print('word2index: ', input_lang.word2index)
# print('word2count: ', input_lang.word2count)
# print('index2word: ', input_lang.index2word)
# print('n_words: ', input_lang.n_words)
#
# print('\ninput_lang attrubites:\n')
# print('word2index: ', output_lang.word2index)
# print('word2count: ', output_lang.word2count)
# print('index2word: ', output_lang.index2word)
# print('n_words: ', output_lang.n_words)

print('Loading models ... \n')

hidden_size = 256

encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
encoder1.load_state_dict(torch.load('models\My models\Model files Seq2Seq\encoder1.pt'))
encoder1.eval()

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)  # load decoder model
attn_decoder1.load_state_dict(torch.load('models\My models\Model files Seq2Seq\\attn_decoder1.pt'))
attn_decoder1.eval()

# Checking state dicts for saving
print("encoder1's state_dict:")
for param_tensor in encoder1.state_dict():
    print(param_tensor, "\t", encoder1.state_dict()[param_tensor].size())

print('\n')
# Checking state dicts for saving
print("attn_decoder1's state_dict:")
for param_tensor in attn_decoder1.state_dict():
    print(param_tensor, "\t", attn_decoder1.state_dict()[param_tensor].size())


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


output_words, attentions = evaluate(
    encoder1, attn_decoder1, "Let's go home.")

print('input =', "Let's go home.")
print('output =', ' '.join(output_words))

# plt.matshow(attentions.numpy()) show plot in streamlit
