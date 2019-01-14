import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QInputDialog, QLineEdit, QFileDialog, QErrorMessage
from PyQt5 import QtCore, QtWidgets, uic
# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#numerical stuff
import pandas as pd
import numpy as np
import itertools as it
#sklearn stuff
from sklearn import manifold
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import euclidean_distances

#make the UI
uiFile = 'naryvizui.ui' # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(uiFile)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        #fix the menubar for osx
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        #UI things go here ....

        #Menubar actions
        self.actionComposition.triggered.connect(self.importFcn)
        self.actionComposition_with_color_code.triggered.connect(self.importFcn)

        #create a canvas ...
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayoutPlotting.addWidget(self.toolbar)
        self.verticalLayoutPlotting.addWidget(self.canvas)
        #make the buttons functional
        self.pushButtonGenerate.clicked.connect(self.onClickGenerate)
        self.pushButtonCalculate.clicked.connect(self.onClickCalculate)
        self.pushButtonReplot.clicked.connect(self.plotPos)

        #set defaults
        self.dim = '2D'
        self.numElements = 5
        self.numSteps = 7
        self.progress = 0
        self.pushButtonReplot.setEnabled(False)
        self.pushButtonCalculate.setEnabled(False)

    def floattest(self,value):
        try:
            float(value)
        except ValueError:
            return False
        else:
            return True

    def importFcn(self, value):
        self.statusBar().showMessage('Importing composition...')

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Excel Files (*.xlsx);;All Files (*)", options=options)
        self.pdframe = pd.read_excel(fileName)
        #treat colors
        sender = self.sender()
        if sender.text() == 'Composition with color code':
            self.statusBar().showMessage('Importing composition with color code...')
            if self.floattest(self.pdframe['ID'].values[0]):
                self.color_value = np.array(self.pdframe['ID'].values)
                self.classes = 1
                self.class_labels = 'Numeric'
            else:
                self.classes = self.pdframe['ID'].drop_duplicates()
                self.class_labels = self.pdframe['ID']
                #messy fix for turning phase labels (text) into numbers
                self.color_value = np.empty(len(self.class_labels))
                for i in range(len(self.class_labels)):
                    for j in range(len(self.classes)):
                        if self.class_labels[i] == self.classes.iloc[j]:
                            self.color_value[i] = j
        self.compsMeasured = self.pdframe.drop('ID',axis=1).values
        #update progress
        self.progress += 30
        self.progressBar.setValue(self.progress)
        self.pushButtonCalculate.setEnabled(True)


    def onClickGenerate(self,value):
        self.statusBar().showMessage('Generating...')
        self.genComp(n=self.spinBoxNumSteps.value(),inary=self.spinBoxNumElements.value())
        self.progress += 40
        self.progressBar.setValue(self.progress)
        self.pushButtonCalculate.setEnabled(True)

    def onClickCalculate(self,value):
        self.statusBar().showMessage('Calculate...')
        if hasattr(self,'compsGenerated') and hasattr(self,'pdframe'):

            #ex data in comps generated frame
            combined = np.append(self.compsMeasured,self.compsGenerated,axis=0)
            print(len(combined))
            self.gen_mds_coords(combined)
            self.plotPos()
        elif hasattr(self,'compsGenerated'):
            #efor lurking
            self.gen_mds_coords(self.compsGenerated)
            self.plotPos()
        elif hasattr(self,'pdframe'):
            #efor lurking
            self.gen_mds_coords(self.pdframe.drop('ID',axis=1).values)
            self.plotPos()

        self.progress = 100
        self.progressBar.setValue(self.progress)


    def genComp(self,n=7,inary=4):
        el = np.array([i/n for i in range(n+1)])
        _comps = np.array([x for x in it.product(el, repeat=inary) if np.isclose(np.sum(x),1)])*100
        self.statusBar().showMessage('Generated {}-nary in {} steps gave {} compositions'.format(inary,n,len(_comps)))
        self.compsGenerated = _comps

    def gen_mds_coords(self,compo):
        #first decide the dimensionality ...
        if self.comboBoxDim.currentText() == '2D':
            dim = 2
        else:
            dim = 3
        #check the desired precision
        if self.comboBoxPrecision.currentText() == 'low':
            max_iter = 6000
            eps = 1e-9

        elif self.comboBoxPrecision.currentText() == 'medium':
            max_iter = 6000
            eps = 1e-10
        elif self.comboBoxPrecision.currentText() == 'high':
            max_iter = 18000
            eps = 1e-20

        self.statusBar().showMessage('MDS with max_iter: {} eps: {}'.format(max_iter,eps))
        #then do the mds accordingly
        self.compo = compo
        self.compo -= self.compo.mean()
        similarities = euclidean_distances(self.compo)
        seed = np.random.RandomState(seed=3) #for repeatability

        mds = manifold.MDS(n_components=dim, max_iter=max_iter, eps=eps, random_state=seed,
                           dissimilarity='precomputed')
        print('Similarities have shape {}'.format(np.shape(similarities)))
        pos = mds.fit(similarities).embedding_
        pos *= np.sqrt((self.compo ** 2).sum()) / np.sqrt((self.compo ** 2).sum())
        #print(np.shape(pos))
        self.pos = pos
        self.plotPos()

    def plotPos(self):
        self.pushButtonReplot.setEnabled(True)
        self.figure.clear()
        cmap = self.comboBoxCmap.currentText()
        if self.comboBoxDim.currentText() == '2D':
            ax = self.figure.add_subplot(111)
            ax.axis('equal')
            if hasattr(self,'color_value'):
                l = len(self.compsMeasured)
                self.cpos =  self.pos[0:l,:]
                sc = ax.scatter(self.cpos[:,0],self.cpos[:,1],s=50, c=np.array(self.color_value),edgecolor='black',cmap=cmap)
                if self.class_labels != 'Numeric':
                    cbar = plt.colorbar(sc,ax=ax, ticks=[i for i in range(len(self.classes))])
                    cbar.ax.set_yticklabels(self.classes)
                else:
                    cbar = plt.colorbar(sc,ax=ax)
                ax.scatter(self.pos[l+1:,0],self.pos[l+1:,1], s=20, alpha=0.1)
                #draw labels
                if self.checkBoxLabels.isChecked():
                    for label,i in zip(self.pdframe.columns.values,range(len(self.pdframe.columns.values)-1)):
                        xy = self.pos[np.argmax(self.compo[:,i]),:]
                        ax.text(xy[0], xy[1], label)
            else:
                ax.scatter(self.pos[:,0],self.pos[:,1])
            ax.axis('off')
        #this case is just for lurking at the composition maps
        elif self.comboBoxDim.currentText() == '3D':
            ax = self.figure.add_subplot(111, projection = '3d')
            ax.axis('equal')
            if hasattr(self,'color_value'):
                l = len(self.compsMeasured)
                self.cpos =  self.pos[0:l,:]
                sc = ax.scatter(self.cpos[:,0],self.cpos[:,1],self.cpos[:,2],s=50, c=np.array(self.color_value),edgecolor='black',cmap=cmap)
                if self.class_labels != 'Numeric':
                    cbar = plt.colorbar(sc,ax=ax, ticks=[i for i in range(len(self.classes))])
                    cbar.ax.set_yticklabels(self.classes)
                else:
                    cbar = plt.colorbar(sc,ax=ax)
                ax.scatter(self.pos[l+1:,0],self.pos[l+1:,1],self.pos[l+1:,2], s=20, alpha=0.1)
                if self.checkBoxLabels.isChecked():
                    for label,i in zip(self.pdframe.columns.values,range(len(self.pdframe.columns.values)-1)):
                        xy = self.pos[np.argmax(self.compo[:,i]),:]
                        ax.text(xy[0], xy[1], xy[2], label)
            else:
                ax.scatter(self.pos[:,0],self.pos[:,1],self.pos[:,2])
            #ax.scatter(to_plot[:,0],to_plot[:,1],to_plot[:,2],s=50,c=color_valueb,cmap='Accent',edgecolor='black')
            ax.axis('off')
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
