class DummyVectorizer:
    def transform(self, texts):
        return [[1 if 'beta' in t else 0] for t in texts]


class DummyModel:
    def predict_proba(self, X):
        probs=[]
        for row in X:
            if row[0]==1:
                probs.append([0.2,0.8])
            else:
                probs.append([0.8,0.2])
        return probs


class DummyEncoder:
    def __init__(self):
        self.classes_=['CP_A','CP_B']
    def inverse_transform(self, idxs):
        return [self.classes_[i] for i in idxs]
