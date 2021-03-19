from ipywidgets import interact, interactive, IntSlider, Layout, interact_manual
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import FileUpload

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df=pd.read_csv('mortgage-dataset.csv', index_col=[0])
X = df[df.columns.difference(['approved','name'])]
X = pd.get_dummies(X, drop_first=True)
y = df['approved']

def func_fit(model_type,test_size,depth,features):
        
    X1 = X[list(features)]
    
    if (model_type=='tree'):
        tree = DecisionTreeClassifier(max_depth=depth)
        tree.fit(X1,y)
        fig = plt.figure(figsize = (10,5), dpi=900)
        plot_tree(tree, feature_names=X1.columns, filled=True)
        pred = tree.predict(X1)
        accuracy = accuracy_score(pred, y)
        transparency = 'High'
        
    if (model_type=='forest'):
        rf = RandomForestClassifier(max_depth=depth)
        rf.fit(X1,y)
        fig, axes = plt.subplots(nrows = 1,ncols = 4,figsize = (10,5), dpi=900)
        for index in range(0, 4):
            plot_tree(rf.estimators_[index],feature_names = X1.columns,class_names=['successful','not successful'],filled = True,ax = axes[index]);
        pred = rf.predict(X1)
        accuracy = accuracy_score(pred, y)
        transparency = 'Low'
        
    print('Accuracy = ', accuracy)
    print('Transparency = ', transparency)
    return

def interactive_tree_model():
    style = {'description_width': 'initial'}
    # Continuous_update = False for IntSlider control to stop continuous model evaluation while the slider is being dragged
    m = interactive(func_fit,model_type=widgets.RadioButtons(options=['tree','forest'],
                                                        description = "Choose Model",style=style,
                                                        layout=Layout(width='250px')),
                    test_size=widgets.Dropdown(options={"10% of data":0.1,"20% of data":0.2, "30% of data":0.3,
                                                        "40% of data":0.4,"50% of data":0.5},
                                              description="Test set size ($X_{test}$)",style=style),
                    depth=widgets.IntSlider(value=3, min=1,max=10,step=1,description= 'Tree depth',
                                           style=style,continuous_update=False),
                    features = widgets.SelectMultiple(options=X.columns,value=list(X.columns),
                                                      description='Variables',disabled=False))

    # Set the height of the control.children[-1] so that the output does not jump and flicker
    output = m.children[-1]
    output.layout.height = '1000px'

    # Display the control
    return display(m)