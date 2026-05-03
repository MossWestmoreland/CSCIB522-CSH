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

    "NoStarving": ("Nostarving.txt", 1),
    "PortaitoftheArtist": ("Dedalus.txt", 1),
    "BeachWeak": ("Bweak.txt", 1),
    "ForeignSong": ("Foreign.txt", 1),
    "Psst": ("Psst.txt", 1),
    "SunHot": ("Sunhot.txt", 1),
    "BeachFagz": ("Bfagz.txt", 1),
    "SummerBummer": ("Bummer.txt", 1),
    "RyanNorthbyNorthWest": ("RyanNorth.txt", 1),
    "BeachDrugs": ("Bdrugs.txt", 1),
    "BeachDeath": ("Bdeath.txt", 1),
    "Beatles": ("Beatles.txt", 1),

    "Tybee": ("Tybee.txt", 1),
    "Sunday": ("Sunday.txt", 1),
    "Somnambulist": ("Somnambulist.txt", 1),
    "Happyugly": ("Happyugly.txt", 1),
    "Upallnight": ("Upallnight.txt", 1),
    "Customer": ("Customer.txt", 1),
    "Racist": ("Racist.txt", 1),
    "Kidwar": ("Kidwar.txt", 1),
    "College": ("College.txt", 1),
    "Bulletin": ("Bulletin.txt", 1),
    "Veterans": ("Veterans.txt", 1),
    "Mydad": ("Mydad.txt", 1),
    "OUJ": ("OUJ.txt", 1),

    "Smokezone": ("Smokezone.txt", 1),
    "Coffehouse": ("Coffeehouse.txt", 1),
    "InSpace": ("InSpace.txt", 1),
    "Asshole": ("Asshole.txt", 1),
    "Shoelaces": ("Shoelaces.txt", 1),
    "Suspicious": ("Suspicious.txt", 1),
    "Reversejacket": ("Reversejacket.txt", 1),
    "Majestic": ("Majestic.txt", 1),
    "90": ("90.txt", 1),
    "Fiction": ("Fiction.txt", 1),
    "Womensapparel": ("Womensapparel.txt", 1),
    "Sameasearth": ("Sameasearth.txt", 1),

    "Bridge2cross": ("Bridge2cross.txt", 1),
    "whoevenknows": ("whoevenknows.txt", 1),
    "Thewhoknows": ("Thewhoknows.txt", 1),
    "Kiddingaround": ("Kiddingaround.txt", 1),
    "Heartless": ("Heartless.txt", 1),
    "dickless": ("dickless.txt", 1),
    "danieljohnston": ("danieljohnston.txt", 1),
    "bobsaget": ("bobsaget.txt", 1),
    "Around": ("Around.txt", 1),

    "Drum": ("Drum.txt", 1),
    "newsforsadness": ("newsforsadness.txt", 1),
    "stoopkid": ("stoopkid.txt", 1),
    "Surfjerk": ("Surfjerk.txt", 1),
    "Somethingsoon": ("Somethingsoon.txt", 1),
    "Passion": ("Passion.txt", 1),
    "Strangers": ("Strangers.txt", 1),
    "Terror": ("Terror.txt", 1),
    "Lawn": ("Lawn.txt", 1),
    "Eyesshut": ("Eyesshut.txt", 1),
    "openmouthedboy": ("openmouthedboy.txt", 1),

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

    "Sylvia": ("Sylvia.txt", 1),
    "weather": ("weather.txt", 1),
    "Disneyworld": ("Disneyworld.txt", 1),
    "math": ("math.txt", 1),
    "AC": ("AC.txt", 1),
    "wherelove": ("wherelove.txt", 1),
    "mrpilot": ("mrpilot.txt", 1),
    "witch": ("witch.txt", 1),
    "psychotic": ("psychotic.txt", 1),
    "clowns": ("clowns.txt", 1),

    "boxing": ("boxing.txt", 1),
    "2years": ("2years.txt", 1),
    "salutary": ("salutary.txt", 1),
    "kara": ("kara.txt", 1),
    "Douchy": ("Douchy.txt", 1),
    "eighth": ("eighth.txt", 1),
    "street": ("street.txt", 1),
    "Sumner": ("Sumner.txt", 1),
    "Nemesis": ("Nemesis.txt", 1),
    "Spicy": ("Spicy.txt", 1),
    "Mannequin": ("Mannequin.txt", 1),
    "na": ("na.txt", 1),
    "trainlane": ("trainlane.txt", 1),

    "left": ("left.txt", 1),
    "Head": ("Head.txt", 1),
    "224": ("224.txt", 1),
    "Cl": ("Cl.txt", 1),
    "Williamsburg": ("Williamsburg.txt", 1),
    "memory": ("memory.txt", 1),
    "Cicis": ("Cicis.txt", 1),
    "Bushy": ("Bushy.txt", 1),
    "pomegranate": ("pomegranate.txt", 1),
    "Doodle": ("Doodle.txt", 1),

    "Time": ("Time.txt", 1),
    "Trapped": ("Trapped.txt", 1),
    "806": ("806.txt", 1),
    "Seine": ("Seine.txt", 1),
    "Walked": ("Walked.txt", 1),
    "ISF": ("ISF.txt", 1),
    "Eyeball": ("Eyeball.txt", 1),
    "Bigjacket": ("Bigjacket.txt", 1),
    "Fine": ("Fine.txt", 1),

    "Napoleon": ("Napoleon.txt", 1),
    "Nite.txt": ("Nite.txt", 1),
    "Die": ("Die.txt", 1),
    "Vicodin": ("Vicodin.txt", 1),
    "Hazelnut": ("Hazelnut.txt", 1),
    "Skyscraper": ("Skyscraper.txt", 1),
    "Bignose": ("Bignose.txt", 1)
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