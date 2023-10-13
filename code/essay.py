import pandas as pd
import re
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class Essay:

    def __init__(self, file, text, **kwargs):
        self.file = file
        self.text = text
        self.paragraphs = kwargs.get('paragraphs', self.get_paragraphs(text))
        self.title = kwargs.get('title', None)
        self.annotation = kwargs.get('annotation', None)
        self.paragraph_labels = kwargs.get('paragraph_labels', [])
        self.title_embeddings = kwargs.get('title_embeddings', None)
        self.ling_errors = kwargs.get('ling_errors', [])
        self.argumentative = kwargs.get('argumentative', None)

    def __len__(self):
        # length = [len(para.doc) for para in self.paragraphs]
        # length = sum(length)
        # if length == 0:
        #     print("Length equals zero")
        length = sum([len(para) for para in self.paragraphs])
        return length



    def get_paragraphs(self, text):
        paragraphs = []
        text = re.split(r"\n+", text.rstrip("\n"))
        #print(text)
        for index, t in enumerate(text):
            paragraphs.append(Paragraph(file=self.file, no=index, text=t))
        return paragraphs

    def __str__(self):
        return str(self.__dict__)

    def to_para_df(self):
        records = []
        for para in self.paragraphs:
            records.append(para.__dict__)
        return pd.DataFrame.from_records(records)


class Paragraph:

    def __init__(self, file, no, text, **kwargs):

        self.file = file
        self.no = no
        self.sentences =  kwargs.get('sentences', sent_tokenize(text))

        self.pos_tags = []
        self.tokens = []
        for sent in self.sentences:
            tokens, pos = zip(*nltk.pos_tag(nltk.word_tokenize(sent)))
            self.pos_tags.append(list(pos))
            self.tokens.append(list(tokens))

        self.sent_ranks = kwargs.get('sent_rank', [])


        self.adu_flow = kwargs.get('adu_flow', [])
        self.adu_flow_wo_none = kwargs.get('adu_flow_wo_none', [])
        self.adu_changes_only = kwargs.get('adu_changes_only', [])
        self.adu_changes_wo_none = kwargs.get('adu_changes_wo_none', [])
        self.adu_wo_none_changes = kwargs.get('adu_wo_none_changes', [])

        self.offsets = kwargs.get('offsets', [])
        self.entities = kwargs.get('entities', [])
        self.sentence_labels = kwargs.get('adu_wo_none_changes', [])
        self.sent_scores = kwargs.get('sent_scores', [])
        self.candidates = kwargs.get('candidates', [])
        self.sentiment = kwargs.get('sentiment', [])
        #self.ling_errors = kwargs.get('candidates', [])


        self.paragraph_embeddings = kwargs.get('paragraph_embeddings', None)

    def __repr__(self):
        return str(self.__dict__)

    def __len__(self):
        length = len([item for sublist in self.tokens for item in sublist])
        return length



class Annotation:
    def __init__(self, id, **kwargs):
        self.id_ = id
        self.arg_strength_score = kwargs.get('arg_strength_score', None)
        self.arg_strength_fold =  kwargs.get('arg_strength_fold', None)
        self.organization_score =  kwargs.get('organization_score', None)
        self.organization_fold =  kwargs.get('organization_fold', None)
        self.thesis_clarity_score = kwargs.get('thesis_clarity_score', None)
        self.thesis_clarity_fold = kwargs.get('thesis_clarity_fold', None)

    def __repr__(self):
        return str(self.__dict__)