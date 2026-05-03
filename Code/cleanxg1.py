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
    #8 songs

    "MyBoy": ("Myboy.txt", 1),
    "BeachLifeinDeath": ("BLID.txt", 1),
    "StopSmoking": ("Smoking.txt", 1),
    "Sober": ("Sobertodeath.txt", 1),
    "NervousInhuman": ("NYI.txt", 1),
    "Bodys": ("Bodys.txt", 1),
    "CuteThing": ("cutething.txt", 1),
    "HighDeath": ("Hightodeath.txt", 1),
    "ThoseBoys": ("TwinFantasy.txt", 1),
    "FamousProphets": ("famous prophets.txt", 1),
    #10 songs

    "JoeGoesSchool": ("School.txt", 2),
    "ConnectDots": ("Sinatra.txt", 2),
    "CostaConcordia": ("Concordia.txt", 2),
    "CosmicHero": ("Cosmic.txt", 2),
    "UnforgivingGirl": ("Unforgiving.txt", 2),
    "1937": ("1937.txt", 2),
    "DrunkDrivers": ("Whales.txt", 2),
    "NotWhatINeeded": ("Needed.txt", 2),
    "DrugsWithFriends": ("Drugwfriends.txt", 2),
    "HippiePowers": ("Hippie.txt", 2),
    "Vincent": ("Vincent.txt", 2),
    "FillintheBlank": ("Blanks.txt", 2),
    #12 songs

    "ReusetheCels": ("Cels.txt", 3),
    "IHateLiving": ("Living.txt", 3),
    "ItsOnlySex": ("Sex.txt", 3),
    "DevilMoon": ("Devilmoon.txt", 3),
    #4 songs

    "NoStarving": ("Nostarving.txt", 4),
    "PortaitoftheArtist": ("Dedalus.txt", 4),
    "BeachWeak": ("Bweak.txt", 4),
    "ForeignSong": ("Foreign.txt", 4),
    "Psst": ("Psst.txt", 4),
    "SunHot": ("Sunhot.txt", 4),
    "BeachFagz": ("Bfagz.txt", 4),
    "SummerBummer": ("Bummer.txt", 4),
    "RyanNorthbyNorthWest": ("RyanNorth.txt", 4),
    "BeachDrugs": ("Bdrugs.txt", 4),
    "BeachDeath": ("Bdeath.txt", 4),
    "Beatles": ("Beatles.txt", 4),
    #12 songs
    
    "Tybee": ("Tybee.txt", 5),
    'Sunday': ("Sunday.txt", 5),
    "Somnambulist": ("Somnambulist.txt", 5),
    "Happyugly": ("Happyugly.txt", 5),
    "Upallnight": ("Upallnight.txt", 5),
    "Customer": ("Customer.txt", 5),
    "Racist": ("Racist.txt", 5),
    "Kidwar": ("Kidwar.txt", 5),
    "College": ("College.txt", 5),
    "Bulletin": ("Bulletin.txt", 5),
    "Veterans": ("Veterans.txt", 5),
    "Mydad": ("Mydad.txt", 5),
    "OUJ": ("OUJ.txt", 5),
    #13 songs
    
    "Smokezone": ("Smokezone.txt", 6),
    "Coffehouse": ("Coffeehouse.txt", 6),
    "InSpace": ("InSpace.txt", 6),
    "Asshole": ("Asshole.txt", 6),
    "Shoelaces": ("Shoelaces.txt", 6),
    "Suspicious": ("Suspicious.txt", 6),
    "Reversejacket": ("Reversejacket.txt", 6),
    "Majestic": ("Majestic.txt", 6),
    "90": ("90.txt", 6),
    "Fiction": ("Fiction.txt", 6),
    "Womensapparel": ("Womensapparel.txt", 6),
    "Sameasearth": ("Sameasearth.txt", 6),
    #12 songs
    
    "Bridge2cross": ("Bridge2cross.txt", 7),
    "whoevenknows": ("whoevenknows.txt", 7),
    "Thewhoknows": ("Thewhoknows.txt", 7),
    "Kiddingaround": ("Kiddingaround.txt", 7),
    "Heartless": ("Heartless.txt", 7),
    "dickless": ("dickless.txt", 7),
    "danieljohnston": ("danieljohnston.txt", 7),
    "bobsaget": ("bobsaget.txt", 7),
    "Around": ("Around.txt", 7),
    #9 songs
    
    "Drum": ("Drum.txt", 8),
    "newsforsadness": ("newsforsadness.txt", 8),
    "stoopkid": ("stoopkid.txt", 8),
    "Surfjerk": ("Surfjerk.txt", 8),
    "Somethingsoon": ("Somethingsoon.txt", 8),
    "Passion": ("Passion.txt", 8),
    "Strangers": ("Strangers.txt", 8),
    "Terror": ("Terror.txt", 8),
    "Lawn": ("Lawn.txt", 8),
    "Eyesshut": ("Eyesshut.txt", 8),
    "openmouthedboy": ("openmouthedboy.txt", 8),
    #11 songs
    
    "Romantictheory": ("Romantictheory.txt", 9),
    "Misheard": ("Misheard.txt", 9),
    "times2die": ("times2die.txt", 9),
    "Overexposed": ("Overexposed.txt", 9),
    "Los": ("Los.txt", 9),
    "Souls": ("Souls.txt", 9),
    "Maud": ("Maud.txt", 9),
    "sleepstrange": ("sleepstrange.txt", 9),
    "Anchorite": ("Anchorite.txt", 9),
    #9 songs
    
    "Depression": ("Depression.txt", 10),
    "remindme": ("remindme.txt", 10),
    "Homes": ("Homes.txt", 10),
    "Afterglow": ("Afterglow.txt", 10),
    "Jerks": ("Jerks.txt", 10),
    "Birds": ("Birds.txt", 10),
    "gunsong": ("gunsong.txt", 10),
    "goodbye": ("goodbye.txt", 10),
    "piano": ("piano.txt", 10),
    "crows": ("crows.txt", 10),
    "sweat": ("sweat.txt", 10),
    "burning": ("burning.txt", 10),
    "Dreams": ("Dreams.txt", 10),
    "Plane": ("Plane.txt", 10),
    "movies": ("movies.txt", 10),
    "jus": ("jus.txt", 10),
    "angel": ("angel.txt", 10),
    "knife": ("knife.txt", 10),
    #18 songs
    
    "Weightlifters": ("Weightlifters.txt", 11),
    "coolme": ("coolme.txt", 11),
    "deadlines": ("deadlines.txt", 11),
    "Hollywood": ("Hollywood.txt", 11),
    "hymn": ("hymn.txt", 11),
    "Martin": ("Martin.txt", 11),
    "Thoughtful": ("Thoughtful.txt", 11),
    "Lately": ("Lately.txt", 11),
    "lifeworth": ("lifeworth.txt", 11),
    "blood": ("blood.txt", 11),
    "famous": ("famous.txt", 11),
    #11 songs
    
    "CCF": ("CCF.txt", 12),
    "Dev": ("Dev.txt", 12),
    "Lady": ("Lady.txt", 12),
    "catastrophe": ("catastrophe.txt", 12),
    "equals": ("equals.txt", 12),
    "Gesthemane": ("Gesthemane.txt", 12),
    "reality": ("reality.txt", 12),
    "desperation": ("desperation.txt", 12),
    "truefalse": ("truefalse.txt", 12),
    #9 songs
    
    "Sylvia": ("Sylvia.txt", 13),
    "weather": ("weather.txt", 13),
    "Disneyworld": ("Disneyworld.txt", 13),
    "math": ("math.txt", 13),
    "AC": ("AC.txt", 13),
    "wherelove": ("wherelove.txt", 13),
    "mrpilot": ("mrpilot.txt", 13),
    "witch": ("witch.txt", 13),
    "psychotic": ("psychotic.txt", 13),
    "clowns": ("clowns.txt", 13),
    #10 songs
    
    "boxing": ("boxing.txt", 14),
    "2years": ("2years.txt", 14),
    "salutary": ("salutary.txt", 14),
    "kara": ("kara.txt", 14),
    "Douchy": ("Douchy.txt", 14),
    "eighth": ("eighth.txt", 14),
    "street": ("street.txt", 14),
    "Sumner": ("Sumner.txt", 14),
    "Nemesis": ("Nemesis.txt", 14),
    "Spicy": ("Spicy.txt", 14),
    "Mannequin": ("Mannequin.txt", 14),
    "na": ("na.txt", 14),
    "trainlane": ("trainlane.txt", 14),
    #13 songs
    
    "left": ("left.txt", 15),
    "Head": ("Head.txt", 15),
    "224": ("224.txt", 15),
    "Cl": ("Cl.txt", 15),
    "Williamsburg": ("Williamsburg.txt", 15),
    "memory": ("memory.txt", 15),
    "Cicis": ("Cicis.txt", 15),
    "Bushy": ("Bushy.txt", 15),
    "pomegranate": ("pomegranate.txt", 15),
    "Doodle": ("Doodle.txt", 15),
    #10 songs
    
    "Time": ("Time.txt", 16),
    "Trapped": ("Trapped.txt", 16),
    "806": ("806.txt", 16),
    "Seine": ("Seine.txt", 16),
    "Walked": ("Walked.txt", 16),
    "ISF": ("ISF.txt", 16),
    "Eyeball": ("Eyeball.txt", 16),
    "Bigjacket": ("Bigjacket.txt", 16),
    "Fine": ("Fine.txt", 16),
    #9 songs
    
    "Napoleon": ("Napoleon.txt", 17),
    "Nite.txt": ("Nite.txt", 17),
    "Die": ("Die.txt", 17),
    "Vicodin": ("Vicodin.txt", 17),
    "Hazelnut": ("Hazelnut.txt", 17),
    "Skyscraper": ("Skyscraper.txt", 17),
    "Bignose": ("Bignose.txt", 17)
    #7 songs
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

plt.figure(figsize=(14, 13))
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