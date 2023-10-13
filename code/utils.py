import shutil
import numpy as np
from essay import Essay, Annotation
import glob
import os
import pandas as pd
import jsonpickle

'''
Dictionary of metadiscourse markers taken from Hyland, K. (2018). Metadiscourse: Exploring interaction in writing. Bloomsbury Publishing.

'''

metadiscourse = {
"Code Glosses" : ["-", "(","as a matter of","fact", "called", "defined as", "e.g.", "for example", "for instance",
                  "i mean", "i.e.", "in fact", "in other words", "indeed","known as","namely", "or", "put another way",
                  "say", "specifically", "such as", "that is", "that is to say", "that means", "this means", "viz",
                  "which means"],

'Endorphoric Markers' : ["chapter", "part", "section", "example", "fig.", "figure", "p.", "page", "table", "above",
                         "before","below","earlier","later"],

"Evidentials" : ["cite", "quote", "according to", "cited", "quoted"],


"Frame Marker Sequencing" : ["chapter", "part" ,"section" ,"finally" ,"first" ,"first of all" ,"firstly" ,"last" ,
                             "lastly" ,"listing","next","numbering","following","second","secondly","subsequently",
                             "then","third","thirdly","to begin","to start with"],


"Frame Marker Label" : ["all in all","at this,point","at this stage","by far","for the,moment","in brief",
                        "in,conclusion","in short","in sum","in summary","now","on the","whole","overall","so far",
                        "thus far","to conclude","to repeat","to sum up","to summarize"],

"Frame Marker Announce" : ["chapter","part","section","aim","desire","to focus","goal","intend to","intention",
                           "objective","purpose","seek to","want to","wish to","would like to"],


"Frame Marker Shift Topic" : ["back to","digress","in regard to","move on","now","resume","return to","revisit",
                              "shift to","so","to look more","closely","turn to","well","with regard to"],

"Transition Markers" : ["accordingly","additionally","again","also","alternatively","although","and","as a consequence",
                        "as a result","at the same time","because","besides","but","by contrast","by the same token",
                        "consequently","conversely","equally","even though","further","furthermore","hence","however",
                        "in addition","in contrast","in the same way","leads to","likewise","moreover",
"nevertheless","nonetheless","on the contrary","on the other hand","rather","result in","similarly","since","so","so as to","still","the result is","thereby","therefore","though","thus","whereas","while","yet"],

"Attitude Markers" : ["!","admittedly","agree","agrees","agreed","amazed","amazing","amazingly","appropriate","appropriately","astonished","astonishing","astonishingly","correctly","curious","curiously","desirable","desirably","disappointed","disappointing",
"disappointingly","disagree","disagreed","disagrees","dramatic","dramatically","essential","essentially","even","expected","expectedly","fortunate","fortunately","hopeful","hopefully","important","importantly","inappropriate","inappropriately","interesting",
"interestingly","prefer","preferable","preferably","prefered","remarkable","remarkably","shocked","shocking","shocked","shocking","shockingly","striking","strikingly","surprised","surprising","surprisingly","unbeleiveable","unbeleiveably",
"understandable","understandably","unexpected","unexpectedly","unfortunate","unfortunately","unusual","unusually","usual"],

"Boosters" : ["actually","always","believe","believed","believes","beyond doubt","certain","certainly","clear","clearly","conclusively","decidedly","definite","definitely","demonstrate","demonstrated","demonstrates","doubtless",
"establish","established","evident","evidently","find","finds","found", "in fact","incontestable","incontestably","incontrovertible","incontrovertiably","indeed","indisputable","indisputably","know","known","must","never",
"no doubt","obvious","obviously","of course","prove","proved","proves","realize","realized","realizes","really","show","showed","shown","shows","sure","surely","think","thinks","thought","truly","true","undeniable","undeniably","undisputedly",
"undoubtedly","without" "doubt"],

"Self Mentions" : ["i","we","me","my","our","mine","us","the" "author","the author's","the writer","the writer's"],

"Engagement Markers" : ["?","reader's","add","allow","analyse","apply","arrange","assess","assume","by the way","calculate","choose","classify","compare","connect","consider","consult","contrast","define","demonstrate","determine",
"do not,develop","employ","ensure","estimate","evaluate","find","follow","go","have to","imagine","incidentally","increase","input","insert","integrate","key","let","let us","let's","look at","mark","measure","mount","must","need to","note","notice",
"observe","one's","order","ought","our","pay","picture","prepare","recall","recover","refer","regard","remember","remove","review","see","select","set","should","show","suppose","state","take","think about","think of","turn","us","use","we","you","your"],

"Hedges" : ["about", "almost", "apparent", "apparently", "appear", "appeared", "appears", "approximately", "argue", "argued", "argues", "around", "assume", "assumed", "broadly", "certain amount", "certain extent","certain level",
"claim", "claimed", "claims", "could", "couldn't", "doubt", "doubtful", "essentially", "estimate", "estimated", "fairly", "feel", "feels", "felt", "frequently", "from my perspective", "from our perspective", "from this perspective", "generally", "guess" ,
"indicate", "indicated", "indicates", "in general"," in most cases", "in most instances", "in my opinion", "in my view", "in this view", "in our opinion", "in our view", "largely", "likely", "mainly", "may", "maybe", "might", "mostly", "often", "on the whole",
"ought", "perhaps", "plausible", "plausibly", "possible", "possibly", "postulate", "postulated", "postulates", "presumable", "presumably", "probable", "probably", "quite", "rather", "relatively", "roughly", "seems", "should", "sometimes",
"somewhat", "suggest", "suggested", "suggests", "suppose", "supposed", "supposes", "suspect", "suspects", "tend to", "tended to", "tends to", "to my knowledge", "typical", "typically", "uncertain", "uncertianly", "unclearly", "unlikely", "usually",
"would", "wouldn't"]

}

def get_essay(filename, path="data_essays\\", titlepath=None):

    with open(filename, 'r', encoding="utf8", errors='ignore') as f:
        rows = f.readlines()[1:]
        text = ''.join(rows)
        #print(text)

        filename = filename.replace(path, "")
        filename = filename.replace(".txt", "")

    essay = Essay(id=filename, text=text)

    return essay


def get_essays(path, title_path="GERM_corrected_ICLE_metadata.xls", sheet_name='GERM_corrected', annotations= None, version='v2'):
    essays = []
    if version == 'v2':
        df_metadata = pd.read_excel(title_path, sheet_name=sheet_name, usecols=['file', 'title']).set_index('file')
    elif version == 'v3':
        df_metadata = pd.read_excel(title_path, sheet_name=sheet_name, usecols=['File name', 'Title', 'Type']).set_index('File name')

    for filename in glob.iglob(path + '/*.txt', recursive=True):
                with open(filename, 'r', encoding="utf8", errors='ignore') as f:
                    rows = f.readlines()[1:]

                    if rows[0] == '\n':
                        rows = rows[1:]
                    text = ''.join(rows)


                    filename = filename.replace(path, "")
                    filename = filename.replace(".txt", "")

                essay = Essay(file=filename, text=text)
                if annotations is not None and filename in annotations:
                    essay.annotation = annotations[filename]
                #essay.title = df_metadata.loc[[filename], ["title"]]
                if version == 'v3':
                    #print(filename)
                    essay.title = df_metadata["Title"].loc[filename]
                    argumentative =  df_metadata["Type"].loc[filename]
                    if argumentative == 'Argumentative':
                        essay.argumentative = True
                elif version == 'v2':
                    essay.title = df_metadata["title"].loc[filename]

                if version == 'v3':
                    if essay.argumentative:
                        essays.append(essay)
                else:
                    essays.append(essay)
    return essays


def move_files():
    for filename in glob.iglob("persing" + '/*Scores.txt', recursive=True):
        print(filename)
        annotated_files_df = pd.read_csv(filename, delimiter='\t')
        for index, row in annotated_files_df.iterrows():
            file = row[0] + ".txt"
            shutil.copy('data_all_6085_essays\data_all_6085_essays\\' + file, 'data_essays\\' + file)


def process_annotations():
    essay_annotations = {}
    for filepath in glob.iglob('../data/persing_annotations/*Scores.txt', recursive=True):
        annotated_files_df = pd.read_csv(filepath, delimiter='\t')
        for index, row in annotated_files_df.iterrows():
            #print(row[0])
            if row[0] not in essay_annotations:
                essay_annotations[row[0]] = Annotation(id=row[0])
            essay_ann = essay_annotations[row[0]]
            if "ArgumentStrengthScores" in filepath:
                essay_ann.arg_strength_score = row[1]
            if "OrganizationScores" in filepath:
                essay_ann.organization_score= float(row[1].split(",")[0])
            if "ThesisClarity" in filepath:
                essay_ann.thesis_clarity_score = row[1]
    for filepath in glob.iglob('../data/persing_annotations/*Folds.txt', recursive=True):
        df = pd.read_csv(filepath, delimiter='\t', skip_blank_lines=False, header=None)
        annotated_folds_df = np.split(df, df[df.isnull().all(1)].index)
        for index, df in enumerate(annotated_folds_df):
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            for i, row in df.iterrows():
                if "ArgumentStrength" in filepath:
                    essay_annotations[row[0]].arg_strength_fold = index
                if "Organization" in filepath:
                    essay_annotations[row[0]].organization_fold = index
                if "ThesisClarity" in filepath:
                    essay_annotations[row[0]].thesis_clarity_fold = index

    return essay_annotations

def reset_paragraph_embedding(essay):
    essay.title_embeddings = []
    for para in essay.paragraphs:
        para.paragraph_embeddings = []

def load_essays_json(path="annotated_essays/"):
    essays = []
    for filename in glob.iglob(path + '/*.json', recursive=False):
        with open(filename, 'r') as f:
            json_essay = f.read()
        essay = jsonpickle.decode(json_essay)
        essays.append(essay)
        #print(essays)
    essays = sorted(essays, key=lambda essay: essay.file)
    return essays


def store_essays_json(essays, path="annotated_essays/", overwrite=False, unpickable=True):

    if isinstance(essays, (list,np.ndarray)):
        for essay in essays:
            if overwrite is True:
                #reset_paragraph_embedding(essay)
                json_essay = jsonpickle.encode(essay, indent=4,  unpicklable=unpickable)
                with open(path +str(essay.file) + ".json", "w") as outfile:
                    outfile.write(json_essay)
            elif not os.path.exists(path + str(essay.file) + ".json"):
                #reset_paragraph_embedding(essay)
                json_essay = jsonpickle.encode(essay, indent=4, unpicklable=unpickable)
                with open(path + str(essay.file) + ".json", "w") as outfile:
                    outfile.write(json_essay)
    else:
        if overwrite is True:
            #reset_paragraph_embedding(essays)
            json_essay = jsonpickle.encode(essays, indent=4, unpicklable=unpickable)
            with open(path + str(essays.file) + ".json", "w") as outfile:
                outfile.write(json_essay)
        elif not os.path.exists(path + str(essays.file) + ".json"):
            #reset_paragraph_embedding(essays)
            json_essay = jsonpickle.encode(essays, indent=4, unpicklable=unpickable)
            with open(path + str(essays.file) + ".json", "w") as outfile:
                outfile.write(json_essay)



