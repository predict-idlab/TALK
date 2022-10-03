from ink.base.connectors import StardogConnector
from ink.base.structure import InkExtractor
from ink.miner.rulemining import RuleSetMiner
from rdflib_hdt import HDTStore
from rdflib import URIRef, Graph, Literal
import pandas as pd
from ink.base.connectors import AbstractConnector
import json
import stardog
from tqdm import tqdm
import numpy as np
import six
#import sys
#sys.modules['sklearn.externals.six'] = six
#from skrules import SkopeRules


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

    def get_all_begin_events(self):
        global store
        res = store.hdt_document.search((None, URIRef("http://example.org/isBeginEvent"), None))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities

    def get_all_events_of_type(self, a):
        global store
        res = store.hdt_document.search((None, URIRef("https://saref.etsi.org/core/hasActivity"), Literal(a)))[0]
        entities = set()
        for r in tqdm(res):
            entities.add(r[0].toPython())
        return entities

file = "event_graph_30_seconds.hdt"
store = HDTStore(file)
if __name__ == '__main__':
    connector = HDTConnector()
    extractor = InkExtractor(connector, verbose=True)
    all_activities = connector.get_all_events()
    #specific_activities = connector.get_all_events_of_type("UsingMobilePhone")

    pos = set(list(all_activities))  # set(list(set([x['a']['value'] for x in results['results']['bindings']]))[0:10])#
    neg = None  # set(list(all_activities-pos)[0:10])
    X_train, y_train = extractor.create_dataset(11, pos, neg, jobs=4,
                                                skip_list=['https://saref.etsi.org/core/hasActivity', 'https://saref.etsi.org/core/hasTimestamp', 'http://example.org/hasRoutine'])

    X_train = extractor.fit_transform(X_train,float_rpr=True)
    # df_train = pd.DataFrame.sparse.from_spmatrix(X_train[0], index=X_train[1], columns=X_train[2])

    import pandas as pd

    def func1(data):
        features = data[0].tocsc()
        cols = data[2]
        drops = set()
        for i in tqdm(range(0, features.shape[1])):
            # if 'real_data' not in cols[i]:
            if features[:, i].sum() == 1 or features[:, i].getnnz() == 1:
                drops.add(i)
        n_cols = [j for j, i in tqdm(enumerate(cols)) if j not in drops]
        return features[:, n_cols], X_train[1], [i for j, i in enumerate(cols) if j not in drops]


    print(X_train[0].shape)
    X_train = func1(X_train)
    print(X_train[0].shape)
    # df_train.to_csv('table.csv')
    df_train = pd.DataFrame.sparse.from_spmatrix(X_train[0], index=X_train[1], columns=X_train[2])
    # print('start saving')
    df_train.to_pickle('protego_event_30s_minute_depth11_for_ml.pkl')