import torch
from torch import nn
import numpy as np
from transformers import RobertaModel
# Seeding for deterministic results
RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
HIDDEN_LAYER_UNITS = 128

# CLASS_NAMES = ['support', 'deny', 'query', 'comment']
# CLASS_NAMES = ['None','MajorClaim','Claim','Premise']

RANDOM_SEED = 16
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
MAX_LENGTH = 100
BATCH_SIZE = 4
EPOCHS = 7
HIDDEN_UNITS = 128

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class Tfidf_Nn(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(2255, HIDDEN_LAYER_UNITS)
        # Output layer
        self.output = nn.Linear(HIDDEN_LAYER_UNITS, n_classes)
        self.dropout = nn.Dropout(0.1)

        # Defining tanh activation and softmax output
        self.tanh = nn.Tanh()  # Using tanh as it performed better than ReLu during hyper-param optimisation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of the below operations
        x = self.hidden(x)
        # print(x.shape)
        y = self.tanh(x)
        # print(y.shape)
        z = self.dropout(y)
        # print(z.shape)
        z = self.output(z)
        # print(z.shape)
        z = self.softmax(z)

        # returning the output from hidden layer and the output layer
        return y, z


class ADUClassifier(nn.Module):

    def __init__(self, n_classes):
        super(ADUClassifier, self).__init__()
        self.robertaModel = RobertaModel.from_pretrained('roberta-large')  # use roberta-large or roberta-base
        self.model_TFIDF = Tfidf_Nn(n_classes)  # Pre-trained SNN trained with TF-IDF features

        self.drop = nn.Dropout(p=0.3)

        self.input_size_preTrain_tfidf = self.robertaModel.config.hidden_size + HIDDEN_UNITS
        self.out = nn.Linear(self.input_size_preTrain_tfidf, n_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, inputs_tfidf_feats):
        roberta_output = self.robertaModel(
            input_ids=input_ids,  # Input sequence tokens
            attention_mask=attention_mask)  # Mask to avoid performing attention on padding tokens
        # print(roberta_output[1].shape)

        tfidf_hidddenLayer, tfidf_output = self.model_TFIDF(inputs_tfidf_feats)
        # print(tfidf_hidddenLayer.shape)
        # print(tfidf_output.shape)

        # Conactenating pooled output from RoBERTa with the hidden layer from the pre-trained SNN using TF-IDF features.
        # pooled_output = torch.cat((roberta_output[1], tfidf_output) , dim=1)-------- Experimenting with Output of pre-trained SNN
        pooled_output = torch.cat((roberta_output[1], tfidf_hidddenLayer), dim=1)
        output = self.drop(pooled_output)
        output = self.out(output)


        return self.softmax(output)

