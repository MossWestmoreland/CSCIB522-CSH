import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
import umap
import matplotlib.pyplot as plt
from collections import Counter
import textstat
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
#all-mpnet-base-v2 = 0.51959
#multi-qa-mpnet-base-dot-v1 = 0.378026
#all-distilroberta-v1 = 0.4866
#all-MiniLM-L12-v2 = 0.366318
#multi-qa-distilbert-cos-v1 = 0.448176
#all-MiniLM-L6-v2 = 0.392074
#multi-qa-MiniLM-L6-cos-v1 = 0.540828
#paraphrase-multilingual-mpnet-base-v2 = 0.318123
#paraphrase-albert-small-v2 = 0.369116
#paraphrase-multilingual-MiniLM-L12-v2 = 0.4737646
#paraphrase-MiniLM-L3-v2 = 0.43656
#distiluse-base-multilingual-cased-v1 = 0.383812
#distiluse-base-multilingual-cased-v2 = 0.387019


songs = {
    "Dramamine": ("Dramamine.txt", 0),
    "BeastMonster": ("BMT.txt", 0),
    "Kimochi": ("Kimochi.txt", 0),
    "InLove": ("ILWM.txt", 0),
    "America": ("America.txt", 0),
    "IWantYou": ("Awake.txt", 0),
    "Titanic": ("Titaniccsh.txt", 0),
    "SpaceCadet": ("SpaceCadet.txt", 0),

    "MyBoy": ("Myboy.txt", 0),
    "BeachLifeinDeath": ("BLID.txt", 0),
    "StopSmoking": ("Smoking.txt", 0),
    "Sober": ("Sobertodeath.txt", 0),
    "NervousInhuman": ("NYI.txt", 0),
    "Bodys": ("Bodys.txt", 0),
    "CuteThing": ("cutething.txt", 0),
    "HighDeath": ("Hightodeath.txt", 0),
    "ThoseBoys": ("TwinFantasy.txt", 0),
    "FamousProphets": ("famous prophets.txt", 0),

    "JoeGoesSchool": ("School.txt", 0),
    "ConnectDots": ("Sinatra.txt", 0),
    "CostaConcordia": ("Concordia.txt", 0),
    "CosmicHero": ("Cosmic.txt", 0),
    "UnforgivingGirl": ("Unforgiving.txt", 0),
    "1937": ("1937.txt", 0),
    "DrunkDrivers": ("Whales.txt", 0),
    "NotWhatINeeded": ("Needed.txt", 0),
    "DrugsWithFriends": ("Drugwfriends.txt", 0),
    "HippiePowers": ("Hippie.txt", 0),
    "Vincent": ("Vincent.txt", 0),
    "FillintheBlank": ("Blanks.txt", 0),

    "ReusetheCels": ("Cels.txt", 0),
    "IHateLiving": ("Living.txt", 0),
    "ItsOnlySex": ("Sex.txt", 0),
    "DevilMoon": ("Devilmoon.txt", 0),

    "NoStarving": ("Nostarving.txt", 0),
    "PortaitoftheArtist": ("Dedalus.txt", 0),
    "BeachWeak": ("Bweak.txt", 0),
    "ForeignSong": ("Foreign.txt", 0),
    "Psst": ("Psst.txt", 0),
    "SunHot": ("Sunhot.txt", 0),
    "BeachFagz": ("Bfagz.txt", 0),
    "SummerBummer": ("Bummer.txt", 0),
    "RyanNorthbyNorthWest": ("RyanNorth.txt", 0),
    "BeachDrugs": ("Bdrugs.txt", 0),
    "BeachDeath": ("Bdeath.txt", 0),
    "Beatles": ("Beatles.txt", 0),

    "Tybee": ("Tybee.txt", 0),
    "Sunday": ("Sunday.txt", 0),
    "Somnambulist": ("Somnambulist.txt", 0),
    "Happyugly": ("Happyugly.txt", 0),
    "Upallnight": ("Upallnight.txt", 0),
    "Customer": ("Customer.txt", 0),
    "Racist": ("Racist.txt", 0),
    "Kidwar": ("Kidwar.txt", 0),
    "College": ("College.txt", 0),
    "Bulletin": ("Bulletin.txt", 0),
    "Veterans": ("Veterans.txt", 0),
    "Mydad": ("Mydad.txt", 0),
    "OUJ": ("OUJ.txt", 0),

    "Smokezone": ("Smokezone.txt", 0),
    "Coffehouse": ("Coffeehouse.txt", 0),
    "InSpace": ("InSpace.txt", 0),
    "Asshole": ("Asshole.txt", 0),
    "Shoelaces": ("Shoelaces.txt", 0),
    "Suspicious": ("Suspicious.txt", 0),
    "Reversejacket": ("Reversejacket.txt", 0),
    "Majestic": ("Majestic.txt", 0),
    "90": ("90.txt", 0),
    "Fiction": ("Fiction.txt", 0),
    "Womensapparel": ("Womensapparel.txt", 0),
    "Sameasearth": ("Sameasearth.txt", 0),

    "Bridge2cross": ("Bridge2cross.txt", 0),
    "whoevenknows": ("whoevenknows.txt", 0),
    "Thewhoknows": ("Thewhoknows.txt", 0),
    "Kiddingaround": ("Kiddingaround.txt", 0),
    "Heartless": ("Heartless.txt", 0),
    "dickless": ("dickless.txt", 0),
    "danieljohnston": ("danieljohnston.txt", 0),
    "bobsaget": ("bobsaget.txt", 0),
    "Around": ("Around.txt", 0),

    "Drum": ("Drum.txt", 0),
    "newsforsadness": ("newsforsadness.txt", 0),
    "stoopkid": ("stoopkid.txt", 0),
    "Surfjerk": ("Surfjerk.txt", 0),
    "Somethingsoon": ("Somethingsoon.txt", 0),
    "Passion": ("Passion.txt", 0),
    "Strangers": ("Strangers.txt", 0),
    "Terror": ("Terror.txt", 0),
    "Lawn": ("Lawn.txt", 0),
    "Eyesshut": ("Eyesshut.txt", 0),
    "openmouthedboy": ("openmouthedboy.txt", 0),

    "Romantictheory": ("Romantictheory.txt", 0),
    "Misheard": ("Misheard.txt", 0),
    "times2die": ("times2die.txt", 0),
    "Overexposed": ("Overexposed.txt", 0),
    "Los": ("Los.txt", 0),
    "Souls": ("Souls.txt", 0),
    "Maud": ("Maud.txt", 0),
    "sleepstrange": ("sleepstrange.txt", 0),
    "Anchorite": ("Anchorite.txt", 0),

    "Depression": ("Depression.txt", 0),
    "remindme": ("remindme.txt", 0),
    "Homes": ("Homes.txt", 0),
    "Afterglow": ("Afterglow.txt", 0),
    "Jerks": ("Jerks.txt", 0),
    "Birds": ("Birds.txt", 0),
    "gunsong": ("gunsong.txt", 0),
    "goodbye": ("goodbye.txt", 0),
    "piano": ("piano.txt", 0),
    "crows": ("crows.txt", 0),
    "sweat": ("sweat.txt", 0),
    "burning": ("burning.txt", 0),
    "Dreams": ("Dreams.txt", 0),
    "Plane": ("Plane.txt", 0),
    "movies": ("movies.txt", 0),
    "jus": ("jus.txt", 0),
    "angel": ("angel.txt", 0),
    "knife": ("knife.txt", 0),

    "Weightlifters": ("Weightlifters.txt", 0),
    "coolme": ("coolme.txt", 0),
    "deadlines": ("deadlines.txt", 0),
    "Hollywood": ("Hollywood.txt", 0),
    "hymn": ("hymn.txt", 0),
    "Martin": ("Martin.txt", 0),
    "Thoughtful": ("Thoughtful.txt", 0),
    "Lately": ("Lately.txt", 0),
    "lifeworth": ("lifeworth.txt", 0),
    "blood": ("blood.txt", 0),
    "famous": ("famous.txt", 0),

    "CCF": ("CCF.txt", 0),
    "Dev": ("Dev.txt", 0),
    "Lady": ("Lady.txt", 0),
    "catastrophe": ("catastrophe.txt", 0),
    "equals": ("equals.txt", 0),
    "Gesthemane": ("Gesthemane.txt", 0),
    "reality": ("reality.txt", 0),
    "desperation": ("desperation.txt", 0),
    "truefalse": ("truefalse.txt", 0),

    "Sylvia": ("Sylvia.txt", 0),
    "weather": ("weather.txt", 0),
    "Disneyworld": ("Disneyworld.txt", 0),
    "math": ("math.txt", 0),
    "AC": ("AC.txt", 0),
    "wherelove": ("wherelove.txt", 0),
    "mrpilot": ("mrpilot.txt", 0),
    "witch": ("witch.txt", 0),
    "psychotic": ("psychotic.txt", 0),
    "clowns": ("clowns.txt", 0),

    "boxing": ("boxing.txt", 0),
    "2years": ("2years.txt", 0),
    "salutary": ("salutary.txt", 0),
    "kara": ("kara.txt", 0),
    "Douchy": ("Douchy.txt", 0),
    "eighth": ("eighth.txt", 0),
    "street": ("street.txt", 0),
    "Sumner": ("Sumner.txt", 0),
    "Nemesis": ("Nemesis.txt", 0),
    "Spicy": ("Spicy.txt", 0),
    "Mannequin": ("Mannequin.txt", 0),
    "na": ("na.txt", 0),
    "trainlane": ("trainlane.txt", 0),

    "left": ("left.txt", 0),
    "Head": ("Head.txt", 0),
    "224": ("224.txt", 0),
    "Cl": ("Cl.txt", 0),
    "Williamsburg": ("Williamsburg.txt", 0),
    "memory": ("memory.txt", 0),
    "Cicis": ("Cicis.txt", 0),
    "Bushy": ("Bushy.txt", 0),
    "pomegranate": ("pomegranate.txt", 0),
    "Doodle": ("Doodle.txt", 0),

    "Time": ("Time.txt", 0),
    "Trapped": ("Trapped.txt", 0),
    "806": ("806.txt", 0),
    "Seine": ("Seine.txt", 0),
    "Walked": ("Walked.txt", 0),
    "ISF": ("ISF.txt", 0),
    "Eyeball": ("Eyeball.txt", 0),
    "Bigjacket": ("Bigjacket.txt", 0),
    "Fine": ("Fine.txt", 0),

    "Napoleon": ("Napoleon.txt", 0),
    "Nite.txt": ("Nite.txt", 0),
    "Die": ("Die.txt", 0),
    "Vicodin": ("Vicodin.txt", 0),
    "Hazelnut": ("Hazelnut.txt", 0),
    "Skyscraper": ("Skyscraper.txt", 0),
    "Bignose": ("Bignose.txt", 0),
    
    "Ham1": ("Ham1.txt", 1),
    "Ham2": ("Ham2.txt", 1),
    "Ham3": ("Ham3.txt", 1),
    "Ham4": ("Ham4.txt", 1),
    "Ham5": ("Ham5.txt", 1),
    
    "A2S1": ("A2S1.txt", 1),
    "A2S2": ("A2S2.txt", 1),
    
    "A3S1": ("A3S1.txt", 1),
    "A3S2": ("A3S2.txt", 1),
    "A3S3": ("A3S3.txt", 1),
    "A3S4": ("A3S4.txt", 1),
    
    "A4S1": ("A4S1.txt", 1),
    "A4S2": ("A4S2.txt", 1),
    "A4S3": ("A4S3.txt", 1),
    "A4S4": ("A4S4.txt", 1),
    "A4S5": ("A4S5.txt", 1),
    "A4S6": ("A4S6.txt", 1),
    "A4S7": ("A4S7.txt", 1),
    
    "A5S1": ("A5S1.txt", 1),
    "A5S2": ("A5S2.txt", 1),
    
    "TBK1": ("TBK1.txt", 2),
    "TBK2": ("TBK2.txt", 2),
    "TBK3": ("TBK3.txt", 2),
    "TBK4": ("TBK4.txt", 2),
    "TBK5": ("TBK5.txt", 2),
    
    "B2C1": ("B2C1.txt", 2),
    "B2C2": ("B2C2.txt", 2),
    "B2C3": ("B2C3.txt", 2),
    "B2C4": ("B2C4.txt", 2),
    "B2C5": ("B2C5.txt", 2),
    "B2C6": ("B2C6.txt", 2),
    "B2C7": ("B2C7.txt", 2),
    "B2C8": ("B2C8.txt", 2),
    
    "B3C1": ("B3C1.txt", 2),
    "B3C2": ("B3C2.txt", 2),
    "B3C3": ("B3C3.txt", 2),
    "B3C4": ("B3C4.txt", 2),
    "B3C5": ("B3C5.txt", 2),
    "b3c6": ("b3c6.txt", 2),
    "B3C7": ("B3C7.txt", 2),
    "B3C8": ("B3C8.txt", 2),
    "B3C9": ("B3C9.txt", 2),
    "B3C10": ("B3C10.txt", 2),
    "B3C11": ("B3C11.txt", 2)
}

vector = []
classification = []

for name, (path, album) in songs.items(): #gets the text
    with open(path, "r") as f: #r = read, f = file
        lyrics = f.read()

    lines = [l.strip() for l in lyrics.split("\n") if l.strip()] #cleaning lines
    words = re.findall(r"\b\w+\b", lyrics.lower()) #tokenization

    word_count = len(words) #sum of words
    unique_word_count = len(set(words)) #sum of unique words
    ttr = len(set(words)) / len(words) if words else 0.0

    linesum = len(lines)
    linelen = [len(re.findall(r"\b\w+\b", l.lower())) for l in lines]
    avglinelen = np.mean(linelen) if linelen else 0.0
    stdlinelen = np.std(linelen) if linelen else 0.0

    punc = re.findall(r"[.,!?;:]", lyrics)
    puncden = len(punc) / len(words) if words else 0.0

    counts = Counter(lines)
    replines = sum(1 for l, c in counts.items() if c > 1)
    chorusrep = max(counts.values()) if counts else 0
    uniqueline = len(counts) / len(lines) if lines else 0.0

    endings = []
    for line in lines: #Tries to find typical one line rhymes
        w = re.findall(r"\b\w+\b", line.lower())
        if w:
            last = w[-1]
            endings.append(last[-2:] if len(last) >= 2 else last)

    if endings:
        end_counts = Counter(endings)
        rhymed = sum(c for e, c in end_counts.items() if c > 1)
        rhyme_dens = rhymed / len(endings)
    else:
        rhyme_dens = 0.0

    try:
        fog = textstat.gunning_fog(lyrics)
    except:
        fog = 0.0

    features = np.array([
        word_count,
        unique_word_count,
        ttr,
        linesum,
        avglinelen,
        stdlinelen,
        puncden,
        replines,
        chorusrep,
        uniqueline,
        rhyme_dens,
        fog
    ], dtype=float)

    emb = model.encode(lyrics, convert_to_numpy=True)
    fullvector = np.concatenate([emb, features])

    vector.append(fullvector)
    classification.append(album)

X = np.array(vector)
y = np.array(classification)


scaler = StandardScaler()
Xscaled = scaler.fit_transform(X)

umap3d = umap.UMAP(
    n_components=15,
    n_neighbors=3,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

Xumap = umap3d.fit_transform(Xscaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect(None, zoom=0.9)

scatter = ax.scatter(
    Xumap[:, 0],
    Xumap[:, 1],
    Xumap[:, 2],
    c=y,
    cmap="tab10",
    s=40
)

ax.set_title("Album classification", fontsize=20)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

legend = ax.legend(*scatter.legend_elements(), title="Group")
plt.show()

unique_labels = sorted(set(y))
avg = []

for label in unique_labels:
    avg.append(Xscaled[y == label].mean(axis=0))

avg = np.array(avg)
corr_matrix = np.corrcoef(avg)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Correlation Between Albums")
plt.show()

xg = XGBClassifier(
    n_estimators=600,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=len(set(y)),
    eval_metric="mlogloss"
)

kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

scores = cross_val_score(xg, Xscaled, y, cv=kf, scoring="accuracy")
y_pred = cross_val_predict(xg, Xscaled, y, cv=kf)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

f1_scores = []
for train_idx, test_idx in kf.split(Xscaled, y):
    xg.fit(Xscaled[train_idx], y[train_idx])
    preds = xg.predict(Xscaled[test_idx])
    f1_scores.append(f1_score(y[test_idx], preds, average="macro"))

print("F1 mean:", np.mean(f1_scores))
print("F1 std:", np.std(f1_scores))

print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))
std_acc = np.std(scores)
print("Std deviation:", std_acc)
errors = 1 - scores
rmse = np.sqrt(np.mean(errors ** 2))
print("RMSE of fold errors:", rmse)

'''
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
'''