import pickle

import numpy as np
import pandas as pd
import scipy
import torch
# Seeding for deterministic results
import transformers
from sentence_transformers import SentenceTransformer
from torch.utils import data
from models import ADUClassifier

RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
HIDDEN_LAYER_UNITS = 128
MAX_LENGTH = 100
BATCH_SIZE = 256
# CLASS_NAMES = ['support', 'deny', 'query', 'comment']
# CLASS_NAMES = ['None','MajorClaim','Claim','Premise']
CLASS_NAMES = ['Non-ADU', 'ADU']

tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-large')
# open a file, where you stored the pickled data
file = open('../models/tfidf.pickle', 'rb')

# dump information to that file
tfidf = pickle.load(file)

if torch.cuda.is_available():
    # set device to GPU
    device = torch.device("cuda")
    # print('There are %d GPU(s) available.' % torch.cuda.device_count())
    # print('We will use the GPU:', torch.cuda.get_device_name(0))

# If no GPU is available
else:
    # print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# Converting labels to numbers
def label_to_int(label):
    if label == 0:
        return 0
    elif label == 1:
        return 1
    elif label == 2:
        return 1
    elif label == 3:
        return 1


# Converting labels to numbers
def adu_nonadu(label):
    if label == 'None':
        return 'Non-ADU'
    elif label == 'MajorClaim':
        return 'ADU'
    elif label == 'Claim':
        return 'ADU'
    elif label == 'Premise':
        return 'ADU'


# Creates a dataset which will be used to feed to RoBERTa
class IcleDataset(data.Dataset):
    def __init__(self, TextSrcInre, tokenizer, max_len):
        self.TextSrcInre = TextSrcInre  # Concatenation of reply+ previous+ src text to get features from 1 training example
        self.tokenizer = tokenizer  # tokenizer that will be used to tokenize input sequences (Uses BERT-tokenizer here)
        self.max_len = max_len  # Maximum length of the tokens from the input sequence that BERT needs to attend to

    def __len__(self):
        return len(self.TextSrcInre)

    def __getitem__(self, item):

        TextSrcInre = str(self.TextSrcInre[item])


        # Encoding the first and the second sequence to a form accepted by RoBERTa
        # RoBERTa does not use token_type_ids to distinguish the first sequence from the second sequnece.
        encoding = tokenizer.encode_plus(
            TextSrcInre,
            max_length=self.max_len,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'TextSrcInre': TextSrcInre,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def createIcleDataLoader(dataframe, tokenizer, max_len, batch_size):
    ds = IcleDataset(
        TextSrcInre=dataframe.Text.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2  # 2
    )


def get_predictions(model, data_loader):
    model.to(device)
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []

    with torch.no_grad():
        for d in data_loader:
            textSrcInre = d["TextSrcInre"]

            #
            input_ids = d["input_ids"].to(device)  #
            attention_mask = d["attention_mask"].to(device)  #
            tfidf_transform = tfidf.transform(textSrcInre)  #
            tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(tfidf_transform)).float()

            tfidf_transform_tensor = tfidf_transform_tensor.to(device)

            # Getting the softmax output from model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_tfidf_feats=tfidf_transform_tensor

            )
            _, preds = torch.max(outputs, dim=1)  # Determining the model predictions

            sentences.extend(textSrcInre)
            predictions.extend(preds)
            prediction_probs.extend(outputs)


    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()

    return sentences, predictions, prediction_probs


def int_to_label_adu(label):
    if label == 0:
        return 'Non-ADU'
    elif label == 1:
        return 'ADU'


def int_to_label_types(label):
    if label == 0:
        return 'MajorClaim'
    elif label == 1:
        return 'Claim'
    elif label == 2:
        return 'Premise'


def predict_single_sent(model, sentence):
    model.to(device)
    model = model.eval()
    encoding = tokenizer.encode_plus(
        sentence,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    # print(encoding)
    d = {
        'TextSrcInre': sentence,
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }
    with torch.no_grad():
        textSrcInre = d["TextSrcInre"]
        #
        input_ids = d["input_ids"].to(device)
        #
        attention_mask = d["attention_mask"].to(device)
        # Features            = d["Features"]
        tfidf_transform = tfidf.transform(textSrcInre)



        tfidf_transform_tensor = torch.tensor(scipy.sparse.csr_matrix.todense(tfidf_transform)).float()

        tfidf_transform_tensor = tfidf_transform_tensor.to(device)

        # Getting the softmax output from model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_tfidf_feats=tfidf_transform_tensor
        )
        _, preds = torch.max(outputs, dim=1)
    #


def predict_single_ADU(sentence):
    CLASS_NAMES = ['Non-ADU', 'ADU']
    model = ADUClassifier(len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/ADU_NonADU_NoFeatures_params.pt"), map_location=device)
    predict_single_sent(model, sentence)


def predict_ADU_Non(model, dataset):
    ds = IcleDataset(TextSrcInre=dataset.Text.to_numpy(), tokenizer=tokenizer, max_len=MAX_LENGTH)
    ICLE_DataLoader = data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    sentences, icle_labels, predProbs_icle = get_predictions(model, ICLE_DataLoader)
    #
    predictions = pd.DataFrame(list(icle_labels.numpy()), columns=['Ann_Adu'])

    predictions['Ann_Adu'] = predictions.Ann_Adu.apply(int_to_label_adu)
    predictions['Ann_Adu'].value_counts()
    return pd.concat([dataset.reset_index(drop=True), predictions], axis=1)


def predict_ADU(model, dataset):
    icle_adu_df = dataset.loc[dataset['Ann_Adu'] != 'Non-ADU']
    icle_adu_df['index1'] = icle_adu_df.index

    ds = IcleDataset(TextSrcInre=icle_adu_df.Text.to_numpy(), tokenizer=tokenizer, max_len=MAX_LENGTH)
    ICLE_DataLoader = data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    sentences, icle_labels, predProbs_icle = get_predictions(model, ICLE_DataLoader)


    index_1 = pd.DataFrame(icle_adu_df[['index1', 'file']])
    index_1 = index_1.reset_index(drop=True)

    predictions = pd.DataFrame(list(zip(sentences, icle_labels.numpy())), columns=['Text', 'ADU_Type'])
    concatenated_dataframes = pd.concat([index_1, predictions], axis=1, ignore_index=True)
    concatenated_dataframes.columns = ['index', 'file', 'Text', 'Ann_Adu']
    concatenated_dataframes['Ann_Adu'] = concatenated_dataframes.Ann_Adu.apply(int_to_label_types)


    icle_non_adu_df = dataset.loc[dataset['Ann_Adu'] == 'Non-ADU']
    icle_non_adu_df = icle_non_adu_df.reset_index()
    icle_non_adu_df['Ann_Adu'] = 'None'
    icle_non_adu_df = icle_non_adu_df.reset_index(drop=True)
    result = concatenated_dataframes.append(icle_non_adu_df, ignore_index=True)
    result = result.sort_values(by='index')
    result.rename(columns={'Ann_Adu': 'ADU'})

    return result


sBert = None


def embedding_encoding(texts):
    global sBert
    if sBert is None:
        sBert = SentenceTransformer('all-MiniLM-L6-v2')
    return sBert.encode(texts, convert_to_tensor=True)
