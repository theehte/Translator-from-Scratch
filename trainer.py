from Transformer_model import Transformer # this is the Transformer_model.py file
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader



class TextDataset(Dataset):

    def __init__(self, english_sentences, kannada_sentences):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.hindi_sentences[idx]

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space


NEG_INFTY = -1e9

def create_masks(eng_batch, hindi_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, hindi_sentence_length = len(eng_batch[idx]), len(hindi_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      hindi_chars_to_padding_mask = np.arange(hindi_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, hindi_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, hindi_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, hindi_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

english_file = 'train.en'
hindi_file = 'train.hindi' 


  

START_TOKEN = ''
PADDING_TOKEN = ''
END_TOKEN = ''

kannada_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ', 
                     अ, आ, इ, ई, उ, ऊ, ऋ, ए, ऐ, ओ, औ,क्ष, त्र, ज्ञ,ड़, ढ़, श्र,क, ख, ग, घ, ङ, च, छ, ज, झ, ञ, ट, ठ, ड, ढ,
                     ण, त, थ, द, ध, न, प, फ, ब, भ, म, य, र, ल, व, श, ष, स, ह,०,१,२,३,४,५,६,७,८,९,PADDING_TOKEN, END_TOKEN]

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        ':', '<', '=', '>', '?', '@',
                        '[', '\', ']', '^', '_', '`', 
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 
                        '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]


index_to_hindi = {k:v for k,v in enumerate(hindi_vocabulary)}
hindi_to_index = {v:k for k,v in enumerate(hindi_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}

with open(english_file, 'r') as file:
    english_sentences = file.readlines()
with open(hindi_file, 'r') as file:
    hindi_sentences = file.readlines()

# Limit Number of sentences
TOTAL_SENTENCES = 200000
english_sentences = english_sentences[:TOTAL_SENTENCES]
hindi_sentences = hindi_sentences[:TOTAL_SENTENCES]
english_sentences = [sentence.rstrip('\n').lower() for sentence in english_sentences]
hindi_sentences = [sentence.rstrip('\n') for sentence in hindi_sentences]
PERCENTILE = 97

max_sequence_length = 200


valid_sentence_indicies = []
for index in range(len(kannada_sentences)):
    hindi_sentence, english_sentence = hindi_sentences[index], english_sentences[index]
    if is_valid_length(hindi_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(hindi_sentence, hindi_vocabulary):
        valid_sentence_indicies.append(index)


d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
hindi_vocab_size = len(hindi_vocabulary)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          hindi_vocab_size,
                          english_to_index,
                          hindi_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN)


hindi_sentences = [hindi_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]
dataset = TextDataset(english_sentences, kannada_sentences)


train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break
     
criterian = nn.CrossEntropyLoss(ignore_index=kannada_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, hindi_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, hindi_batch)
        optim.zero_grad()
        kn_predictions = transformer(eng_batch,
                                     hindi_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(hindi_batch, start_token=False, end_token=True)
        loss = criterian(
            hindi_predictions.view(-1, hindi_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == hindi_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"hindi Translation: {hindi_batch[0]}")
            hindi_sentence_predicted = torch.argmax(hindi_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in hindi_sentence_predicted:
              if idx == hindi_to_index[END_TOKEN]:
                break
              predicted_sentence += index_to_hindi[idx.item()]
            print(f"hindi Prediction: {predicted_sentence}")


            transformer.eval()
            hindi_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, hindi_sentence)
                predictions = transformer(eng_sentence,
                                          hindi_sentence,
                                          encoder_self_attention_mask.to(device), 
                                          decoder_self_attention_mask.to(device), 
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_hindi[next_token_index]
                hindi_sentence = (hindi_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break
            
            print(f"Evaluation translation (should we go to the mall?) : {hindi_sentence}")
            print("-------------------------------------------")

