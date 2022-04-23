from datetime import tzinfo
from itertools import count
from tokenize import Ignore
import pandas as pd
import matplotlib.pyplot as plt
from dateutil.parser import parse
from dateutil.tz import gettz
from time import strptime
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from scipy.stats import spearmanr
tzinfos = gettz('Europe / Berlin')

df = pd.read_csv('Code\Data\ODI-2022.csv', header = 0, sep = ';')
ds = df.describe(include='all')
print(ds)

# df = df.drop([24, 120, 141, 148, 176, 211, 223, 234, 249, 293]) # faulty bdays
# df = df.drop([2, 10, 12, 15, 21, 25, 16]) # faulty bedtimes
#df.iloc[:, 8] = [parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True) if not ParserError else df.drop(df.index[df.iloc[:, 8] == x].tolist, inplace=True) for x in df.iloc[:, 8]]
#df.iloc[:, 14] = [parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True) for x in df.iloc[:, 14]]
# df.iloc[:, 8] = [parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True) for x in df.iloc[:, 8]]
#df.iloc[:, 14] = df.iloc[:, 14].time()
#for x in df.iloc[:, 8]:
#       parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
"""def bdayparse():
    for x in df.iloc[:, 8]:
        if x is ParserError:
            df.drop(x)
        else:
            df.iloc[:, 8] = parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
            return    
"""      
"""for x in df.iloc[:, 8]:
    try:
        parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
    except Exception:
        df.drop(df.index[df.iloc[:, 8] == x], inplace=True)
        # df = df.reset_index(drop=True)
    else:
        break"""
"""for x in df.iloc[:, 8]:
    
    x = parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
    df.iloc[:, 8] = x
# bdayparse()"""
"""try:
    for x in df.iloc[:, 8]:
        parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
except Exception:
    
    df.drop(df.index[df.iloc[:, 8] == x], inplace=True)"""
    
"""try:
    df.iloc[:, 8] = [parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True) for x in df.iloc[:, 8]]
except Exception:
    None
        # df = df.reset_index(drop=True)"""
   
# parsing dates
def myparser(x):
    try:
       return parse(x, tzinfos = tzinfos, dayfirst = True, fuzzy = True)
    except:
       return None
#df.iloc[:, 8]=df.iloc[:, 8].apply(lambda x: myparser(x))
df.iloc[:, 14]=df.iloc[:, 14].apply(lambda x: myparser(x))
"""def extractdate():
    try:
        for x in df.iloc[:, 8]:
            datetime.date(x)
        return 
    except:
        return None"""

#drop created NaN
df.dropna(axis = 0, how = 'any', inplace = True)

#turn datetime to date
#df.iloc[:, 8]=df.iloc[:, 8].apply(lambda x: datetime.date(x))

#turn datetime to time    
#df.iloc[:, 14]=df.iloc[:, 14].apply(lambda x: datetime.time(x))

print(df.iloc[:, 8][0])
print(df.iloc[:, 14][0])
print(df)

# dict of the value counts
def valcountdict(col):
    values = df.iloc[:, col].value_counts(dropna=False).keys().tolist()
    counts = df.iloc[:, col].value_counts(dropna=False).tolist()
    value_dict = dict(zip(values, counts))
    return value_dict
for i in range(16):  
    var = valcountdict([i])
    # print(var)

#df.iloc[:, 1] = df.iloc[:, 1].str.replace('AI','Artificial Intelligence')
# REPLACE ALL AI 
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"AI","AI ", "MSc in AI", "AI Masters", "VU Master of AI", "Master AI","artificial intelligence", 
"M Artificial Intelligence", "Masters in Artificial Intelligence", "Msc AI @ UvA", "Msc Artificial Intelligence @ UvA", "MSc Artificial Intelligence",
 "Ai", "M Artificial Intelligence "}, value="Artificial Intelligence")
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"Artificial Intelligence Cognitive Science Track", "Artificial intelligence master", "Artificial Intelligence Masters", "Artificial Intelligence for health",
   "Artificial Intelligence at UvA", "MSc Artificial Intelligence", "Msc Artificial Intelligence"
   }, value="Artificial Intelligence")
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={ "MSc in Artificial Intelligence", "Artificial Intelligence at UvA",
"Master Artificial Intelligence", "Masters Artificial Intelligence", "MSc in Artificial Intelligence", "Master Artificial intelligence", "Master Artificial Intelligence at UvA"
, "MSc AI", "Msc AI", "MSc. AI at UvA", "MSc. AI at UvA ", "", ""}, value="Artificial Intelligence")
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"MSc. Artificial Intelligence ", "VU Master of Artificial Intelligence", 
"Artificial Intelligence ", "Masters in Artificial Intelligence", "MSc Artificial Intelligence", "Artificial intelligence",
 "Ai", "Masters AI ", "MSc AI at UVA", "AI for health", "AI Cognitive Science Track"}, value="Artificial Intelligence")

# REPLACE ALL Computer Science
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"CS", "Computer Science ", "computer science", "Computer science", "cs", "MSc Computer Science"
, "MSc Computer science", "Master of Computer science", "Computer science - BDE" , "Computer Science(joint degree)", "computer scienece", "CS Big Data Engineering", "Master of Computer science", "", "", ""}, value = "Computer Science")

# REPLACE ALL Business Analytics
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"BA", "business analytics", "Business Analytics Master (Computational Intelligence track)", "Business analytics", ""}, value = "Business Analytics")

# REPLACE ALL Computational Science
df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={"computational science", "Computational science", "Computational Science ", 
"Computational Science(CLS)", "MSc Computational Science", "CLS","MSc Computational Science", "Computational  Science", "", "", ""}, value = "Computational Science")

# REPLACE REST
#df.iloc[:, 1] = df.iloc[:, 1].replace(to_replace={not ("Artificial Intelligence", "Computer Science", "Business Analytics", "Computational Science")}, value = "Other")
df.iloc[:, 1] = np.where(df.iloc[:, 1].isin(["Artificial Intelligence", "Computer Science", "Business Analytics", "Computational Science"]), df.iloc[:, 1], 'Other')

#some descriptives
ds = df.describe(include='all')
print(ds)
df.to_csv('Code\Data\preprocessed-ODI-2022.csv')

# the amount of a certain value in a column
var0 = df.iloc[:, 0].value_counts(); var1 = df.iloc[:, 1].value_counts(); var2 = df.iloc[:, 2].value_counts(); var3 = df.iloc[:, 3].value_counts(); 
var4 = df.iloc[:, 4].value_counts(); var5 = df.iloc[:, 5].value_counts(); var6 = df.iloc[:, 6].value_counts(); var7 = df.iloc[:, 7].value_counts(); 
var8 = df.iloc[:, 8].value_counts(); var9 = df.iloc[:, 9].value_counts(); var10 = df.iloc[:, 10].value_counts(); var11 = df.iloc[:, 11].value_counts()
var12 = df.iloc[:, 12].value_counts(); var13 = df.iloc[:, 13].value_counts(); var14 = df.iloc[:, 14].value_counts(); var15 = df.iloc[:, 15].value_counts(); 
var16 = df.iloc[:, 16].value_counts()

#df.groupby(['team']).sum().plot(kind='pie', y='points')


histprog = df.iloc[:, 1]
histprog.hist(bins=5)
plt.ylabel('Frequency')
#plt.xlabel("Artificial Intelligence, Computer Science, Business Analytics, Computational Science, Other")
plt.title('What programme are you in?')
plt.style.use('seaborn-whitegrid')

plt.show()

histml = df['Have you taken a course on machine learning?']
histml.hist(bins=3)
plt.ylabel('Frequency')
#plt.xlabel('No, Yes, Unknown')
plt.title('Have you taken a course on machine learning?')
plt.style.use('seaborn-whitegrid')
plt.show()

histir = df.iloc[:, 3]
histir.hist(bins=3)
plt.ylabel('Frequency')
#plt.xlabel('No = 0, Yes = 1, Unknown')
plt.title('Have you taken a course on information retrieval')
plt.style.use('seaborn-whitegrid')
plt.show()


histst = df.iloc[:, 4]
histst.hist(bins=3)
plt.ylabel('Frequency')
#plt.xlabel('No = Sigma, Yes = Mu , Unknown')
plt.title('Have you taken a course on statistics')
plt.style.use('seaborn-whitegrid')
plt.show()


histdb = df.iloc[:, 5]
histdb.hist(bins=3)
plt.ylabel('Frequency')
#plt.xlabel('No = Nee, Yes = Ja, Unknown')
plt.title('Have you taken a course on databases?')
plt.style.use('seaborn-whitegrid')
plt.show()


histgen = df.iloc[:, 6]
histgen.hist(bins=5)
plt.ylabel('Frequency')
#plt.xlabel('Male, Female, Unknown')
plt.title('What is your gender?')
plt.style.use('seaborn-whitegrid')
plt.show()


#encoding all the data
df['eattml'] = LabelEncoder().fit_transform(df.iloc[:, 2])
df['eattir'] = LabelEncoder().fit_transform(df.iloc[:, 3])
df['eattst'] = LabelEncoder().fit_transform(df.iloc[:, 4])
df['eattdb'] = LabelEncoder().fit_transform(df.iloc[:, 5])
df['eattgen'] = LabelEncoder().fit_transform(df.iloc[:, 6])
df['etarget'] = LabelEncoder().fit_transform(df.iloc[:, 1])
#creat input and output
target = df['etarget']
atts = df[['eattml', 'eattir' ,'eattst' ,'eattdb' ]] # ,'eattgen'

# splitting data into test and train set
attstrain,attstest, targettrain,targettest = train_test_split(atts, target, test_size=0.33, random_state=42)

"""attstrain = atts[:129]
attstest  = atts[129:]

targettrain = df['etarget'][:129]
targettest  = df['etarget'][129:]

#attstrain, attstest = attstrain.reshape(1,-1), attstest.reshape """ 

kn = KNeighborsClassifier()
kn.fit(attstrain, targettrain)
kn_pred = kn.score(attstest, targettest)
print(f'KN predicts {kn_pred}% correctly.')
knpredictions = kn.predict(attstest)

mlp = MLPClassifier(max_iter=5000)
mlp.fit(attstrain, targettrain)
mlp_pred = mlp.score(attstest, targettest)
print(f'MLP predicts {mlp_pred}% correctly.')
mlppredictions = mlp.predict(attstest)

dt = DecisionTreeClassifier()
dt.fit(attstrain, targettrain)
dt_pred = dt.score(attstest, targettest)
print(f'Decision Tree predicts {dt_pred}% correctly.')
dtpredictions = dt.predict(attstest)

def report_to_latex_table(data):
    avg_split = False
    out = ""
    out += "\\begin{table}\n"
    out += "\\caption{Latex Table from Classification Report}\n"
    out += "\\label{table:classification:report}\n"
    out += "\\centering\n"
    out += "\\begin{tabular}{c | c c c r}\n"
    out += "Class & Precision & Recall & F-score & Support\\\\\n"
    out += "\midrule\n"
    for cls, scores in data.items():
        if 'micro' in cls:
            out += "\\midrule\n"
        out += cls + " & " + " & ".join([str(s) for s in scores])
        out += "\\\\\n"
    out += "\\end{tabular}\n"
    out += "\\end{table}"
    return out


"""from sklearn.metrics import classification_report
print(report_to_latex_table(classification_report(targettest, knpredictions)))
print(report_to_latex_table(classification_report(targettest, mlppredictions)))
print(report_to_latex_table(classification_report(targettest, dtpredictions)))"""



def stressreplace(df):
    df["Stress"] = df['What is your stress level (0-100)?']
    
    for i in df["Stress"]:
        if i.isdigit():
            if float(i) > 100 :
                df["Stress"] = df["Stress"].replace(i,'100')
        else:
            df["Stress"] = df["Stress"].replace(i,'100')


    df["Stress"] = pd.to_numeric(df["Stress"], errors='coerce', downcast='integer')
        
    return df

stressreplace(df)
print()


"""#df.iloc['Bedtime'] = df.iloc[:, 14].apply(lambda x: x.strftime('%H:%M:%S')) # :%S'
df.iloc[:, 14] = pd.to_datetime(df.iloc[:, 14], utc=True, errors='coerce')
df['Bedtime'] = df.iloc[:, 14].dt.strftime('%H:%M:%S')
#[(d - arbitrary_date).total_seconds() for d in dateValues], saleNumbers)
#df['Bedtime'] = df['Bedtime'].apply(lambda x: x[:2])
print('stress: mean=%.3f stdv=%.3f' % (np.mean(df["Stress"]), np.std(df["Stress"])))
print('bedtime: mean=%.3f stdv=%.3f' % (np.mean(df['Bedtime']), np.std(df['Bedtime'])))
# plot
plt.scatter(data1, data2)
plt.show()
"""
def correlations():
    # Data clean up & check
    df.iloc[:, 14] = pd.to_datetime(df.iloc[:, 14], utc=True, errors='coerce')
    df['Bedtime'] = df.iloc[:, 14].dt.strftime('%H:%M:%S')
    df.dropna(axis = 0, how = 'any', inplace = True)
    print('Sample size after deleted missing values=%.3f' % df["Bedtime"].count())
    df['Bedtime'] = df['Bedtime'].apply(lambda x: x[:2])
    df['Bedtime'] = df['Bedtime'].astype(int)
    print('stress: mean=%.3f stdv=%.3f' % (np.mean(df["Stress"]), np.std(df["Stress"])))
    print('bedtime: mean=%.3f stdv=%.3f' % (np.mean(df['Bedtime']), np.std(df['Bedtime'])))
    # Correlation between stress and bedtimes
    sbcorr, _ = pearsonr(df["Stress"], df["Bedtime"])
    print('Stress x Bedtime Pearson R: %.3f' % sbcorr)
    sbscorr, _ = spearmanr(df["Stress"], df["Bedtime"])
    print('Stress x Bedtime Spearman R: %.3f' % sbscorr)
    
    #Correlation between stress and programme
    spcorr, _ = pearsonr(df["Stress"], df["etarget"])
    print('Stress x Programme Pearson R: %.3f' % spcorr)
    spscorr, _ = spearmanr(df["Stress"], df["etarget"])
    print('Stress x Bedtime Spearman R: %.3f' % spscorr)

    #Correlation between stress and gender
    sgcorr, _ = pearsonr(df["Stress"], df["eattgen"])
    print('Stress x Gender Pearson R: %.3f' % sgcorr)
    sgscorr, _ = spearmanr(df["Stress"], df["eattgen"])
    print('Stress x Bedtime Spearman R: %.3f' % sgscorr)

    return 

correlations()


print(df.iloc[:, 1].value_counts())


print('debug end')