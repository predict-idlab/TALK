from catboost import CatBoostClassifier, Pool
from sklearn.multiclass import OutputCodeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
##Using your example data
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GroupShuffleSplit
import numpy as np

from rdflib_hdt import HDTStore
from rdflib import URIRef, Graph, Literal
import pandas as pd
from ink.base.connectors import AbstractConnector
import json
import stardog
from tqdm import tqdm
import seaborn as sns
from mlcm import mlcm
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

class HDTConnector(AbstractConnector):
    def query(self, q_str):
        global store
        try:
            noi = URIRef(q_str.split('"')[1])
            res = store.hdt_document.search((noi, None, None))[0]
            val = [{"p": {"value": r[1].toPython()}, "o": {"value": r[2].n3().split('"')[1]}, "dt": "Object"} if isinstance(r[2],
                                                                                                            Literal) else {
                "p": {"value": r[1].toPython()}, "o": {"value": r[2].toPython()}} for r in res]
            return val
        except Exception as e:
            return []

    def get_all_events(self):
        global store
        res = store.hdt_document.search((None, URIRef("https://saref.etsi.org/core/hasActivity"), None))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities

    def get_all_event_activities(self):
        global store
        #res = store.hdt_document.search((None, URIRef("https://saref.etsi.org/core/hasActivity"), None))[0]
        res = store.hdt_document.search((None, URIRef("https://saref.etsi.org/core/hasActivity"), None))[0]
        entities = set()
        for r in tqdm(res):
            if len(r[2].toPython())>1:
                entities.add((r[0].toPython(), r[2].toPython()))
        return entities

    def get_all_begin_events(self):
        global store
        res = store.hdt_document.search((None, URIRef("http://example.org/isBeginEvent"), None))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities

    def get_all_begin_activities(self):
        global store
        res = store.hdt_document.search((None, URIRef("http://example.org/isBeginEvent"), None))[0]
        entities = set()
        for d in res:
            res2 = store.hdt_document.search((d[0], URIRef("https://saref.etsi.org/core/hasActivity"), None))[0]

            for r in tqdm(res2):
                if len(r[2].toPython())>1:
                    entities.add((d[0].toPython(), r[2].toPython()))
        return entities

    def get_all_events_of_type(self, a):
        global store
        res = store.hdt_document.search((None, URIRef("https://saref.etsi.org/core/hasActivity"), Literal(a)))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities

    def get_event_time(self, user, event):
        global store
        res = store.hdt_document.search((URIRef('https://dahcc.idlab.ugent.be/Protego/'+user+'/event'+event), URIRef("https://saref.etsi.org/core/hasTimestamp"), None))[0]
        return str(list(res)[0][2].toPython())

    def get_prev_label(self, user, event):
        global store
        res = store.hdt_document.search((URIRef('https://dahcc.idlab.ugent.be/Protego/'+user+'/event'+event), URIRef("http://example.org/hasPrevious"), None))[0]
        ex = list(res)[0][2]
        if ex:
            res = store.hdt_document.search((ex,
                                             URIRef("https://saref.etsi.org/core/hasActivity"), None))[0]
            return str(list(res)[0][2].toPython())
        else:
            return "None"

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


def generate_class_weights(class_series, multi_class=True, one_hot_encoded=False):
  """
  Method to generate class weights given a set of multi-class or multi-label labels, both one-hot-encoded or not.

  Some examples of different formats of class_series and their outputs are:
    - generate_class_weights(['mango', 'lemon', 'banana', 'mango'], multi_class=True, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 1.3333333333333333, 'mango': 0.6666666666666666}
    - generate_class_weights([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], multi_class=True, one_hot_encoded=True)
    {0: 0.6666666666666666, 1: 1.3333333333333333, 2: 1.3333333333333333}
    - generate_class_weights([['mango', 'lemon'], ['mango'], ['lemon', 'banana'], ['lemon']], multi_class=False, one_hot_encoded=False)
    {'banana': 1.3333333333333333, 'lemon': 0.4444444444444444, 'mango': 0.6666666666666666}
    - generate_class_weights([[0, 1, 1], [0, 0, 1], [1, 1, 0], [0, 1, 0]], multi_class=False, one_hot_encoded=True)
    {0: 1.3333333333333333, 1: 0.4444444444444444, 2: 0.6666666666666666}

  The output is a dictionary in the format { class_label: class_weight }. In case the input is one hot encoded, the class_label would be index
  of appareance of the label when the dataset was processed.
  In multi_class this is np.unique(class_series) and in multi-label np.unique(np.concatenate(class_series)).

  Author: Angel Igareta (angel@igareta.com)
  """
  if multi_class:
    # If class is one hot encoded, transform to categorical labels to use compute_class_weight
    if one_hot_encoded:
      class_series = np.argmax(class_series, axis=1)

    # Compute class weights with sklearn method
    class_labels = np.unique(class_series)
    class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=class_series)
    return dict(zip(class_labels, class_weights))
  else:
    # It is neccessary that the multi-label values are one-hot encoded
    mlb = None
    if not one_hot_encoded:
      mlb = MultiLabelBinarizer()
      class_series = mlb.fit_transform(class_series)

    n_samples = len(class_series)
    n_classes = len(class_series[0])

    # Count each class frequency
    class_count = [0] * n_classes
    for classes in class_series:
        for index in range(n_classes):
            if classes[index] != 0:
                class_count[index] += 1

    # Compute class weights using balanced method
    class_weights = [n_samples / (n_classes * freq) if freq > 0 else 1 for freq in class_count]
    class_labels = range(len(class_weights)) if mlb is None else mlb.classes_
    return dict(zip(class_labels, class_weights))

#df = pd.read_pickle('protego_event_one_minute_depth6_for_ml.pkl')
df = pd.read_pickle('protego_event_30s_minute_depth3_for_ml.pkl')
print(df.shape)
df = df.T.drop_duplicates().T
print(df.shape)
file = "event_graph_30_seconds.hdt"
store = HDTStore(file)
connector = HDTConnector()

result = connector.get_all_event_activities()
begins = connector.get_all_begin_events()
result = [r for r in result if r[0]]

label_df = pd.DataFrame(result).groupby(0)[1].apply(list)
nodes = label_df.index
labels = label_df.values

#ndata = data.explode('label')
data = df.loc[nodes,[x for x in df.columns if 'Wearable' not in x]]
data.loc[:,'label'] = labels
mask = data.label.apply(lambda x: 'HeartRateMeasurement' in x or 'StudyRelated' in x or 'Unknown' in x)
data = data[~mask]

X = data.drop('label', axis=1)
y = data['label']

from collections import Counter
X = data.drop('label', axis=1)
y = data['label']

with open('columns.txt','w') as f:
    for c in X.columns:
        f.write(c+'\n')

u_set = []
for z in y.values:
    for x in z:
        u_set.append(x)
counts = Counter(u_set)
act_lst =  [w for w,c in counts.most_common(30)]

change = {"EatingMeal":'Eating','EatingSnack':'Eating','Serving':'Eating',
          'RoomTransition':'Neglect','DoorWalkThrough':'Neglect','ObjectUse':'Neglect',
          'Dishwashing':'Organizing',
          'SocialMedia':'UsingMobilePhone','EmailOnMobilePhone':'UsingMobilePhone',
          }
counts['Eating'] += counts['EatingMeal']+counts['EatingSnack']+counts['Serving']
counts['Organizing']+=counts['Dishwashing']
counts['UsingMobilePhone']+=counts['SocialMedia']+counts['EmailOnMobilePhone']

sub_lst = ['Eating']
#act_lst = ["EatingMeal","DrinkPreparation","PreparingMeal","Organizing","Showering","Toileting","UsingComputer","UsingMobilePhone","Walking","WatchingTVActively"]
def select_act(x):
    if len(x)==1:
        if x[0] in change:
            x[0] = change[x[0]]
        if x[0] == 'Neglect':
            return 'Other'
        if x[0] in act_lst or x[0] in sub_lst:
            return x[0]
        else:
            return 'Other'
    else:
        #return 'Other'
        curr_best = 0
        curr_best_res = None
        for el in x:
            if el in change:
                el = change[el]
            if el == 'Neglect':
                continue;
            if counts[el]>curr_best:
                curr_best=counts[el]
                curr_best_res = el
        if curr_best_res in act_lst or curr_best_res in sub_lst:
            return curr_best_res
        else:
            return 'Other'

y = y.apply(lambda y: select_act(y))

u_set = []
for z in y.values:
    u_set.append(z)
counts = Counter(u_set)
act_lst =  [w for w,c in counts.most_common(11)]
print(act_lst)
y = y.apply(lambda x: x if x in act_lst else 'Other')
print(y.value_counts())
act_lst = [z for z in y.unique() ]
total_act_lst = act_lst
print(total_act_lst)
#act_lst = ["EatingMeal","DrinkPreparation","PreparingMeal","Organizing","Showering","Toileting","UsingComputer","UsingMobilePhone","Walking","WatchingTVActively"]
#y = y.apply(lambda y: set([z if z in act_lst else "Other" for z in y]))
#total_act_lst = ["EatingMeal","DrinkPreparation","PreparingMeal","Organizing","Showering","Toileting","UsingComputer","UsingMobilePhone","Walking","WatchingTVActively","Other",""]

#from sklearn.preprocessing import MultiLabelBinarizer
#mlb = MultiLabelBinarizer()
#y = pd.DataFrame(mlb.fit_transform(y),columns=mlb.classes_)
groups = [x.split('/')[-2] for x in X.index]
times = []
get_prev_labels = []
for index, row in X.iterrows():
    user = index.split('/')[-2]
    event = index.split('/')[-1].replace('event','')
    times.append(connector.get_event_time(user, str(event)))
    get_prev_labels.append(connector.get_prev_label(user, str(event)))
#X['time'] = pd.to_datetime(times)
#X['time'] = X['time'].dt.hour * 60 + X['time'].dt.minute + X['time'].dt.second/60

#X['prev_label'] = [act_lst.index(select_act(x)) if select_act(x) in act_lst else -1 for x in get_prev_labels]

oof_pred = []
oof_true = []
group_kfold = GroupKFold(n_splits=30)
best_accuracy = 0
for train_index, test_index in group_kfold.split(X, y, groups):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    group_train, group_test = [groups[i] for i in train_index], [groups[i] for i in test_index]

    splitter = GroupShuffleSplit(test_size=.5, n_splits=1, random_state=42)
    split = splitter.split(X_train, groups=group_train)
    train_inds, eval_inds = next(split)

    X_eval = X_train.iloc[eval_inds, :]
    y_eval = y_train.iloc[eval_inds]
    eval_pool = Pool(X_eval, y_eval)

    X_train = X_train.iloc[train_inds, :]
    y_train = y_train.iloc[train_inds]
    train_pool = Pool(X_train, y_train)

    clf = CatBoostClassifier(
        max_depth=3,
        loss_function='MultiClass',
        eval_metric="TotalF1",
        learning_rate=0.1,
        iterations=2000,
        class_weights=generate_class_weights(y_train.values.tolist() + y_eval.values.tolist(), multi_class=True,
                                             one_hot_encoded=False),
        #auto_class_weights="Balanced"
    )


    clf.fit(train_pool, eval_set=eval_pool, metric_period=10, verbose=100,early_stopping_rounds=100)

    # clf = ExtraTreesClassifier(min_samples_leaf=25, class_weight='balanced', n_jobs=-1)
    # clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    pred_df = pd.DataFrame(y_pred, index=[int(z.split('/')[-1].replace('event', '')) for z in X_test.index])
    oof_pred.append(pred_df.reset_index(drop=True))
    oof_true.append(y_test.reset_index(drop=True))
    accuracy = accuracy_score(y_test.reset_index(drop=True), pred_df.reset_index(drop=True), normalize=True, sample_weight=None)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))



    normal_conf_mat = confusion_matrix(y_test, y_pred, labels=act_lst, normalize='true')

    # df = pd.DataFrame([y_test, y_pred], columns=['y_Actual','y_Predicted'])
    # confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    # conf_mat,normal_conf_mat = mlcm.cm(y_test_multi.reindex(columns=all_columns).fillna(int(0)).values,y_pred_multi.reindex(columns=all_columns).fillna(int(0)).values)
    df_cm = pd.DataFrame(normal_conf_mat, index=act_lst, columns=act_lst )
    #print(df_cm)
    #fig, ax = plt.subplots(figsize=(15, 15))
    #sns.heatmap(df_cm, annot=True, fmt='.2f')
    #plt.savefig('learner_plot_ind_'+str(accuracy)+'_'+group_test[0]+'.png')
    #clf.save_model('model_name'+str(accuracy)+'_'+group_test[0])



from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_pred_multis=[]
for p in oof_pred:
    y_pred_multis.append(p)

y_pred_multi = pd.concat(y_pred_multis)

y_test_multis=[]
for p in oof_true:
    y_test_multis.append(p)
y_test_multi = pd.concat(y_test_multis)

print(y_pred_multi)


from sklearn.metrics import classification_report
print(classification_report(y_test_multi, y_pred_multi))
from sklearn.metrics import confusion_matrix
import numpy as np

normal_conf_mat=confusion_matrix(y_test_multi,y_pred_multi, labels=act_lst, normalize='true')
#normal_conf_mat = normal_conf_mat.astype('float') / normal_conf_mat.sum(axis=1)[:, np.newaxis]
print(normal_conf_mat)

#df = pd.DataFrame([y_test, y_pred], columns=['y_Actual','y_Predicted'])
#confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

#conf_mat,normal_conf_mat = mlcm.cm(y_test_multi.reindex(columns=all_columns).fillna(int(0)).values,y_pred_multi.reindex(columns=all_columns).fillna(int(0)).values)
df_cm = pd.DataFrame(normal_conf_mat, index=act_lst, columns=act_lst)
print(df_cm)
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df_cm, annot=True, fmt='.2f')
plt.savefig('learner_plot.png')