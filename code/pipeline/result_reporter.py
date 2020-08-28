import numpy as np 
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product

# ---------------- Class: ResultReporter ----------------

class ResultReporter:
    '''
        This class is used for managing calculations from a confustion matrix.
    '''
    def __init__(self, confusion_matrix):
        '''
            Initialisation.

            Args:
                confusion_matrix: a quadratic two-dimensional numpy array that is the confusion matrix
        '''
        self.set_confusion_matrix(confusion_matrix)
    
    def set_confusion_matrix(self, confusion_matrix):
        '''
            Resets the confusion matrix of the class.
            Args:
                confusion_matrix: a quadratic numpy array that is the confusion matrix
        '''
        assert confusion_matrix is not None, 'ResultReporter: Confusion Matrix is not allowed to be None.'
        assert type(confusion_matrix).__module__ == 'numpy', 'ResultReporter: Confusion Matrix must be a numpy array.'
        assert len(confusion_matrix.shape) == 2, 'ResultReporter: Confusion Matrix can only be 2-dimensional.'
        assert confusion_matrix.shape[0] == confusion_matrix.shape[1], 'ResultReporter: Confusion Matrix must be quadratic.'
        self.confusion_matrix = confusion_matrix
        self.num_classes = self.confusion_matrix.shape[0]

    def tp(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in
            Returns:
                The number of true positives of the confusion matrix for classification tasks.
        '''
        return self.confusion_matrix[classNum, classNum]

    def fp(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of false positives of the confusion matrix for classification tasks.
        '''
        return self.predPos(classNum) - self.tp(classNum)

    def fn(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of false negatives of the confusion matrix for classification tasks.
        '''
        return self.truthPos(classNum) - self.tp(classNum)

    def tn(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of true negatives of the confusion matrix for classification tasks
        '''
        return self.numSamples() + self.tp(classNum) - self.predPos(classNum) - self.truthPos(classNum)

    def predPos(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of rows (predictions) in a confusion matrix at the classNums column position (truth value) for a classification task
        '''
        return np.sum(self.confusion_matrix[:, classNum])

    def truthPos(self, classNum):
        '''
            Args:
                classNum: The index of the class you are interested in.
            Returns:
                The number of columns (truth values) in a confusion matrix at the classNums row position (prediction) for a classification task
        '''
        return np.sum(self.confusion_matrix[classNum, :])

    def numSamples(self):
        '''
            Returns:
                The overall number of elements (predictions) in a confusion matrix for a classification task
        '''
        return np.sum(self.confusion_matrix)

    def accuracy(self):
        '''
            Returns:
                The accuracy of all predictions of a confusion matrix for a classification task
        '''
        t = np.trace(self.confusion_matrix)
        return t  / self.numSamples()

    def precision(self, classNum):
        '''
            Returns:
                The precision of all predictions of a confusion matrix for a classification task
        '''
        return self.tp(classNum) / max(1, self.predPos(classNum))

    def recall(self, classNum):
        '''
            Returns:
                The recall of all predictions of a confusion matrix for a classification task
        '''
        return self.tp(classNum) / max(1, self.truthPos(classNum))

    def fScore(self, classNum, alpha=1):
        '''
            Args:
                classNum: index of the class you are interested in
                alpha: alpha value
            Returns:
                The class-specific fScore of all predictions of a confusion matrix for a classification task
        '''
        p = self.precision(classNum)
        r = self.recall(classNum)
        if 0 in [p, r]:
            return 0
        return (1 + alpha ** 2) * p * r / (alpha ** 2 * p + r)

    def logFScoreSum(self):
        '''
            Returns:
                The sum of classwise logFScores of all predictions of a confusion matrix for a classification task
        '''
        return np.sum(np.log([max(0.01, self.fScore(classNum)) for classNum in range(self.num_classes)]))

    def logFScoreMean(self):
        '''
            Returns:
                The mean of classwise logFScores of all predictions of a confusion matrix for a classification task
        '''
        return 0.5 * np.mean(np.log10([max(0.01, self.fScore(classNum)) for classNum in range(self.num_classes)])) + 1

    def precisionMacro(self):
        '''
            Returns:
                The mean of classwise precisions of all predictions of a confusion matrix for a classification task
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.precision(classNum) for classNum in range(self.num_classes)])

    def precisionMicro(self):
        '''
            Returns:
                The class-wise-weighted mean of classwise precisions of all predictions of a confusion matrix for a classification task
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        return np.sum([self.tp(classNum) for classNum in range(self.num_classes)]) / np.sum([self.predPos(classNum) for classNum in range(self.num_classes)])

    def recallMacro(self):
        '''
            Returns:
                The mean of classwise recalls of all predictions of a confusion matrix for a classification task
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.recall(classNum) for classNum in range(self.num_classes)])

    def recallMicro(self):
        '''
            Returns:
                The class-wise-weighted mean of classwise recalls of all predictions of a confusion matrix for a classification task
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        return np.sum([self.tp(classNum) for classNum in range(self.num_classes)]) / np.sum([self.truthPos(classNum) for classNum in range(self.num_classes)])

    def fScoreMacro(self, alpha=1):
        '''
            Args:
                alpha: alpha value for fScore computation
            Returns:
                The mean of classwise fScores of all predictions of a confusion matrix for a classification task
                (i.e. metric evaluated independently for each class and then average - hence treating all classes equally)
        '''
        return np.mean([self.fScore(classNum, alpha) for classNum in range(self.num_classes)])

    def fScoreMicro(self, alpha=1):
        '''
            Args:
                alpha: alpha value for fScore computation
            Returns:
                The class-wise-weighted mean of classwise fScores of all predictions of a confusion matrix for a classification task
                (i.e. aggregate the contributions of all classes to compute the average metric)
        '''
        p = self.precisionMicro()
        r = self.recallMicro()
        if p == 0 or r == 0:
            return 0
        return (1 + alpha ** 2) * p * r / (alpha ** 2 * p + r)


    def getResultDict(self):
        d = {
            'logFScoreSum': self.logFScoreSum(),
            'logFScoreMean': self.logFScoreMean(),
            'precisionMacro': self.precisionMacro(),
            'precisionMicro': self.precisionMicro(),
            'recallMacro': self.recallMacro(),
            'recallMicro': self.recallMicro(),
            'accuracy': self.accuracy(),
            'fScoreMacro': self.fScoreMacro(),
            'fScoreMicro': self.fScoreMicro()
        }
        return d

    def plot_confmat(self, sorted_labels, title, folder_path, file_name, save=True, show=False, ax=None, include_values=True, cmap='Blues', xticks_rotation='vertical'):
        ''' Plots the confusion matrix
            # NOTE: adapted version taken from sklearn --> ConfusionMatrixDisplay.plot()

            Args:
                sorted_labels: list of label names that should be shown in the plot (order will follow order given in the list)
                title: the title of the plot (accuracy information will be added in next line)
                folder_path: path to the folder in which to save the confmat plot (including last slash)
                file_name: name of the file ub which to store the plot (without ending)
                save: Boolean to set to True, if plot should be stored
                show (bool): if True, the matrix will be displayed
                ax: an (optional) matplotlib figure axes handle when wanting the confmat plot to be part of a subfigure
                include_values: Boolean to set to True if the values should be included in the plot
                cmap: matplotlib colormap identifier
                xticks_rotation: Rotation of xtick labels, can be {'vertical', 'horizontal'} or float
        '''
        
        assert isinstance(sorted_labels, list), 'ResultReporter: labels for confusion matrix plot must be a list'
        assert len(sorted_labels) == self.num_classes, 'ResultReporter: list of labels for confusion matrix plot must match the number of classes of the confusion matrix.'
        display_labels = np.array(sorted_labels).reshape(-1)

        if ax is None:
            fig, ax = plt.subplots()
            title += '\nAccuracy: {:.2f} %'.format(self.accuracy()*100.0)
            ax.set_title(title)
        else:
            fig = ax.figure


        self.num_classes = self.confusion_matrix.shape[0]
        im_ = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None

        cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

        if include_values:
            text_ = np.empty_like(self.confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (self.confusion_matrix.max() + self.confusion_matrix.min()) / 2.0
            for i, j in product(range(self.num_classes), range(self.num_classes)):
                color = cmap_max if self.confusion_matrix[i, j] < thresh else cmap_min
                text_[i, j] = ax.text(j, i, format(self.confusion_matrix[i, j]), ha="center", va="center", color=color)

        fig.colorbar(im_, ax=ax)
        ax.set(xticks=np.arange(self.num_classes),
               yticks=np.arange(self.num_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="Predicted label",
               xlabel="True label")

        ax.set_ylim((self.num_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
        
        if show:
            plt.show()
        if save:
            save_to = folder_path + file_name + '.png'
            plt.savefig(save_to, format='png', bbox_inches='tight')