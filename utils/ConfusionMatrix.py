import numpy as np
from sklearn.metrics import confusion_matrix

class OnlineConfusionMatrix:
    """
    Calculate binary classification metrics - tp, fp, tn, fn, f1 score.
    Current batch metics can be calculated by calling update_matrix and
    at the end, call to f1_score will provide f1 score
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        
        self.offset = 0.000001

    def update_matrix(self, batch_target, batch_prediction):
        """
        Update current count of true positive, false positive, false negative and true negative
        :param batch_target: 1d numpy array containing binary class label (0/1)
        :param batch_prediction: 1d numpy array containing probability for class being 1
        """
        target = batch_target > 0.5
        prediction = batch_prediction > 0.5

        tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
    
    def f1_score(self) -> float:
        """
        Calculates f1 score as the harmonic mean of precision and recall.
        :return: float 
        """
        precision = self.precision()        
        recall = self.recall()
        f1score = (2 * precision * recall) / (precision + recall)
        return f1score
    
    def precision(self) -> float:
        return (self.tp + self.offset) / (self.tp + self.fp + self.offset)

    def recall(self) -> float:
        return (self.tp + self.offset) / (self.tp + self.fn + self.offset)


    def update_matrix_probability(self, batch_target, batch_prediction):
        """
        Update current count of true positive, false positive, false negative and true negative.
        Method assumes input as [[prob_class_1, prob_class_0], [..]]
        :param batch_target: 2d numpy array 
        :param batch_prediction: 2d numpy array
        """
        
        target = np.argmax(batch_target, axis=1)
        prediction = np.argmax(batch_prediction, axis=1)
        
        # assuming first index in target is positive class and Second indx is neg class
        target = (target + 1) % 2
        prediction = (prediction + 1) % 2
        
        tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
        
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        
if __name__ == "__main__":
    
    cm = OnlineConfusionMatrix() 
    y_true = np.array([1, 0, 0, 1, 1, 0])
    y_pred = np.array([.4, .51, .3, .6, .4, .2])
    cm.update_matrix(y_true, y_pred)
    print("f1 score ", cm.f1_score())
    print(" precision and recall ", cm.precision(), cm.recall())


    from sklearn.metrics import f1_score as fs
    from sklearn.metrics import (precision_score, recall_score)
 
    pr_score = fs(y_true > 0.5, y_pred > 0.5)
    print("sklearn f1 score ", pr_score)
    print("precision ", precision_score(y_true > 0.5, y_pred > 0.5))
    print("recall ", recall_score(y_true > 0.5, y_pred > 0.5))
 



