import numpy as np
import re
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import umap
import matplotlib.pyplot as plt
from collections import Counter
import textstat
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

model = SentenceTransformer("all-distilroberta-v1")
#all-mpnet-base-v2 = 0.43396
#multi-qa-mpnet-base-dot-v1 = 0.35127
#all-distilroberta-v1 = 0.46999
#all-MiniLM-L12-v2 = 0.39815
#multi-qa-distilbert-cos-v1 = 0.39117
#all-MiniLM-L6-v2 = 0.36753
#multi-qa-MiniLM-L6-cos-v1 = 0.44967
#paraphrase-multilingual-mpnet-base-v2 = 0.32716
#paraphrase-albert-small-v2 = 0.344927
#paraphrase-multilingual-MiniLM-L12-v2 = 0.349914
#paraphrase-MiniLM-L3-v2 = 0.338117
#distiluse-base-multilingual-cased-v1 = 0.320624
#distiluse-base-multilingual-cased-v2 = 0.3693028
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

    "JoeGoesSchool": ("School.txt", 1),
    "ConnectDots": ("Sinatra.txt", 1),
    "CostaConcordia": ("Concordia.txt", 1),
    "CosmicHero": ("Cosmic.txt", 1),
    "UnforgivingGirl": ("Unforgiving.txt", 1),
    "1937": ("1937.txt", 1),
    "DrunkDrivers": ("Whales.txt", 1),
    "NotWhatINeeded": ("Needed.txt", 1),
    "DrugsWithFriends": ("Drugwfriends.txt", 1),
    "HippiePowers": ("Hippie.txt", 1),
    "Vincent": ("Vincent.txt", 1),
    "FillintheBlank": ("Blanks.txt", 1),

    "ReusetheCels": ("Cels.txt", 0),
    "IHateLiving": ("Living.txt", 0),
    "ItsOnlySex": ("Sex.txt", 0),
    "DevilMoon": ("Devilmoon.txt", 0),

    "NoStarving": ("Nostarving.txt", 2),
    "PortaitoftheArtist": ("Dedalus.txt", 2),
    "BeachWeak": ("Bweak.txt", 2),
    "ForeignSong": ("Foreign.txt", 2),
    "Psst": ("Psst.txt", 2),
    "SunHot": ("Sunhot.txt", 2),
    "BeachFagz": ("Bfagz.txt", 2),
    "SummerBummer": ("Bummer.txt", 2),
    "RyanNorthbyNorthWest": ("RyanNorth.txt", 2),
    "BeachDrugs": ("Bdrugs.txt", 2),
    "BeachDeath": ("Bdeath.txt", 2),
    "Beatles": ("Beatles.txt", 2),

    "Tybee": ("Tybee.txt", 2),
    "Sunday": ("Sunday.txt", 2),
    "Somnambulist": ("Somnambulist.txt", 2),
    "Happyugly": ("Happyugly.txt", 2),
    "Upallnight": ("Upallnight.txt", 2),
    "Customer": ("Customer.txt", 2),
    "Racist": ("Racist.txt", 2),
    "Kidwar": ("Kidwar.txt", 2),
    "College": ("College.txt", 2),
    "Bulletin": ("Bulletin.txt", 2),
    "Veterans": ("Veterans.txt", 2),
    "Mydad": ("Mydad.txt", 2),
    "OUJ": ("OUJ.txt", 2),

    "Smokezone": ("Smokezone.txt", 2),
    "Coffehouse": ("Coffeehouse.txt", 2),
    "InSpace": ("InSpace.txt", 2),
    "Asshole": ("Asshole.txt", 2),
    "Shoelaces": ("Shoelaces.txt", 2),
    "Suspicious": ("Suspicious.txt", 2),
    "Reversejacket": ("Reversejacket.txt", 2),
    "Majestic": ("Majestic.txt", 2),
    "90": ("90.txt", 2),
    "Fiction": ("Fiction.txt", 2),
    "Womensapparel": ("Womensapparel.txt", 2),
    "Sameasearth": ("Sameasearth.txt", 2),

    "Bridge2cross": ("Bridge2cross.txt", 2),
    "whoevenknows": ("whoevenknows.txt", 2),
    "Thewhoknows": ("Thewhoknows.txt", 2),
    "Kiddingaround": ("Kiddingaround.txt", 2),
    "Heartless": ("Heartless.txt", 2),
    "dickless": ("dickless.txt", 2),
    "danieljohnston": ("danieljohnston.txt", 2),
    "bobsaget": ("bobsaget.txt", 2),
    "Around": ("Around.txt", 2),

    "Drum": ("Drum.txt", 2),
    "newsforsadness": ("newsforsadness.txt", 2),
    "stoopkid": ("stoopkid.txt", 2),
    "Surfjerk": ("Surfjerk.txt", 2),
    "Somethingsoon": ("Somethingsoon.txt", 2),
    "Passion": ("Passion.txt", 2),
    "Strangers": ("Strangers.txt", 2),
    "Terror": ("Terror.txt", 2),
    "Lawn": ("Lawn.txt", 2),
    "Eyesshut": ("Eyesshut.txt", 2),
    "openmouthedboy": ("openmouthedboy.txt", 2),

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

    "Weightlifters": ("Weightlifters.txt", 1),
    "coolme": ("coolme.txt", 1),
    "deadlines": ("deadlines.txt", 1),
    "Hollywood": ("Hollywood.txt", 1),
    "hymn": ("hymn.txt", 1),
    "Martin": ("Martin.txt", 1),
    "Thoughtful": ("Thoughtful.txt", 1),
    "Lately": ("Lately.txt", 1),
    "lifeworth": ("lifeworth.txt", 1),
    "blood": ("blood.txt", 1),
    "famous": ("famous.txt", 1),

    "CCF": ("CCF.txt", 1),
    "Dev": ("Dev.txt", 1),
    "Lady": ("Lady.txt", 1),
    "catastrophe": ("catastrophe.txt", 1),
    "equals": ("equals.txt", 1),
    "Gesthemane": ("Gesthemane.txt", 1),
    "reality": ("reality.txt", 1),
    "desperation": ("desperation.txt", 1),
    "truefalse": ("truefalse.txt", 1),

    "Sylvia": ("Sylvia.txt", 2),
    "weather": ("weather.txt", 2),
    "Disneyworld": ("Disneyworld.txt", 2),
    "math": ("math.txt", 2),
    "AC": ("AC.txt", 2),
    "wherelove": ("wherelove.txt", 2),
    "mrpilot": ("mrpilot.txt", 2),
    "witch": ("witch.txt", 2),
    "psychotic": ("psychotic.txt", 2),
    "clowns": ("clowns.txt", 2),

    "boxing": ("boxing.txt", 2),
    "2years": ("2years.txt", 2),
    "salutary": ("salutary.txt", 2),
    "kara": ("kara.txt", 2),
    "Douchy": ("Douchy.txt", 2),
    "eighth": ("eighth.txt", 2),
    "street": ("street.txt", 2),
    "Sumner": ("Sumner.txt", 2),
    "Nemesis": ("Nemesis.txt", 2),
    "Spicy": ("Spicy.txt", 2),
    "Mannequin": ("Mannequin.txt", 2),
    "na": ("na.txt", 2),
    "trainlane": ("trainlane.txt", 2),

    "left": ("left.txt", 2),
    "Head": ("Head.txt", 2),
    "224": ("224.txt", 2),
    "Cl": ("Cl.txt", 2),
    "Williamsburg": ("Williamsburg.txt", 2),
    "memory": ("memory.txt", 2),
    "Cicis": ("Cicis.txt", 2),
    "Bushy": ("Bushy.txt", 2),
    "pomegranate": ("pomegranate.txt", 2),
    "Doodle": ("Doodle.txt", 2),

    "Time": ("Time.txt", 2),
    "Trapped": ("Trapped.txt", 2),
    "806": ("806.txt", 2),
    "Seine": ("Seine.txt", 2),
    "Walked": ("Walked.txt", 2),
    "ISF": ("ISF.txt", 2),
    "Eyeball": ("Eyeball.txt", 2),
    "Bigjacket": ("Bigjacket.txt", 2),
    "Fine": ("Fine.txt", 2),

    "Napoleon": ("Napoleon.txt", 2),
    "Nite.txt": ("Nite.txt", 2),
    "Die": ("Die.txt", 2),
    "Vicodin": ("Vicodin.txt", 2),
    "Hazelnut": ("Hazelnut.txt", 2),
    "Skyscraper": ("Skyscraper.txt", 2),
    "Bignose": ("Bignose.txt", 2)
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

    meta = np.array([
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
    fullvector = np.concatenate([emb, meta])

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

log = LogisticRegression(max_iter=3000)

kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

scores = cross_val_score(log, Xscaled, y, cv=kf, scoring="accuracy")
y_pred = cross_val_predict(log, Xscaled, y, cv=kf)

cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

f1_scores = []
for train_idx, test_idx in kf.split(Xscaled, y):
    log.fit(Xscaled[train_idx], y[train_idx])
    preds = log.predict(Xscaled[test_idx])
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