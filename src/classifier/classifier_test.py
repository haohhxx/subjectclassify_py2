


class classifier_test(object):
    def __init__(self):
        pass

    def classify(self, X ,y, Xt):
        preall_test = self.use_classify(X, y, Xt)
        # preall_test = clf.predict(Xt)
        return preall_test

    # def classify_proba(self, X , y, Xt):
    #     clf = self.use_classify(X, y)
    #     preall_test = clf.predict_proba(Xt)
    #     return preall_test

    def use_classify(self, X_train, y_train, X_test):
        return ''
