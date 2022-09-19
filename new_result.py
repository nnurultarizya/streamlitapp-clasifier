import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import webbrowser

# Model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# Important
from time import time
import warnings

from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# path to fetch data
data = pd.read_csv('datasetku.csv')
image = Image.open('poltek.png')


# UKT Minimum labels
data = data[data["UKT (Minimum) label"] != 0]
X = data.drop(columns=["program_studi", "get_ukt", "UKT (Minimum) label"]).values
y = data["UKT (Minimum) label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler2 = StandardScaler()
X = scaler2.fit_transform(X)

counter = Counter(y_train)

# compute required values
scaler = StandardScaler().fit(X_train)
train_sc = scaler.transform(X_train)
test_sc = scaler.transform(X_test)


def grafik_actual_vs_predict(pred):
    st.subheader("Prediksi Testing [Actual vs Prediction]")
    hasil= pd.DataFrame(y_test)
    hasil['Prediksi'] = pd.DataFrame(pred)
    st.line_chart(hasil)

def confusion_matrix_plot(x,y):
    conf_mat = confusion_matrix(x, y)
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=["1", "2", "3", "4", "5", "6"]).plot()
    st.pyplot()
    return x,y

def st_write_accuracy(a,b,c,d):
    st.write(""" # Mean Accuracy: *%f* """ %a)
    st.write(""" # Mean Recall: *%f* """ %b)
    st.write(""" # Mean Precision: *%f*  """ %c)
    st.write(""" # Mean F-measure: *%f* """ %d)

def rf() :
    rf = RandomForestClassifier()
    rf.fit(train_sc, y_train)
    rf_pred = rf.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    rf_clf = RandomForestClassifier()
    scores = cross_validate(rf_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    grafik_actual_vs_predict(rf_pred)
    st.write("""
        # Table Predict Random Forest
    """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                    np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    st.subheader("Confusion Matrix Random Forest")
    confusion_matrix_plot(y_test, rf_pred)

def dt() :
    dt = DecisionTreeClassifier()
    dt.fit(train_sc,y_train)
    dt_pred = dt.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    dt_clf = DecisionTreeClassifier()
    scores = cross_validate(dt_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    grafik_actual_vs_predict(dt_pred)

    st.write("""
        # Table Predict Decision Tree""")

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                        np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    st.subheader("Confusion Matrix Decision Tree")
    confusion_matrix_plot(y_test, dt_pred)

def svm():
    svc = SVC()
    svc.fit(train_sc, y_train)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, train_sc, y_train, cv=kfold, scoring=scoring)

    grafik_actual_vs_predict(svc_pred)

    st.write("""
        # Table Predict SVM
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']),np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']),np.mean(scores['test_f1_macro']))

    st.subheader("Confusion Matrix SVM")
    confusion_matrix_plot(y_test, svc_pred)

def pilih_model() :
    pilih_model = st.selectbox('Pilih Model',('Random Forest', 'Decision Tree', 'SVC'))
    if pilih_model == 'Random Forest' :
        rf()
    elif pilih_model == 'Decision Tree' :
        dt()
    elif pilih_model == 'SVC' :
        svm()

def logo_():

    # ----------------------------------- EXPLANATION -----------------------------------
    #     this is what should be simple but quite complicated in streamlit.
    #     streamlit can display an image, but to display an image in the
    #     middle it requires the column() function.
    # ---------------------------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')
    with col2:
        st.image(image)
    with col3:
        st.write(' ')

def smote():
    # button_display()
    oversample = SMOTE()
    X_train_res, y_train_res = oversample.fit_resample(train_sc, y_train)
    st.bar_chart(pd.value_counts(y_train_res))

    time_svm = time()
    svc = SVC()
    svc.fit(X_train_res, y_train_res)
    svc_pred = svc.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    svc_clf = SVC()
    scores = cross_validate(svc_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_svm_calculate= time()-time_svm

    st.write("""
        # Predict SVM SMOTE
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    time_mlp = time()
    mlp = MLPClassifier()
    mlp.fit(X_train_res, y_train_res)
    mlp_pred = mlp.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    mlp_clf = MLPClassifier()
    scores = cross_validate(mlp_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_mlp_calculate= time()-time_mlp


    st.write("""
        # Predict MLP Classifier SMOTE
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    time_rf = time()
    rf = RandomForestClassifier()
    rf.fit(X_train_res, y_train_res)
    rf_pred = rf.predict(test_sc)

    kfold = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    scoring = {'accuracy', 'precision_macro', 'f1_macro', 'recall_macro'}
    rf_clf = RandomForestClassifier()
    scores = cross_validate(rf_clf, X_train_res, y_train_res, cv=kfold, scoring=scoring)
    time_rf_calculate= time()-time_rf


    st.write("""
        # Table Predict Random Forest
     """)

    st_write_accuracy(np.mean(scores['test_accuracy']), np.mean(scores['test_recall_macro']),
                       np.mean(scores['test_precision_macro']), np.mean(scores['test_f1_macro']))

    data_time_calculate =pd.DataFrame(
        {'SVM':[time_svm_calculate],
        'MLP':[time_mlp_calculate],
        'Random Forest':[time_rf_calculate]}
    )

    st.bar_chart(data_time_calculate.loc[0],use_container_width=True)

def bar_bs():
    bar_before_smote = pd.value_counts(data['UKT (Minimum) label'])
    st.header("Label Sebelum SMOTE")
    st.bar_chart(bar_before_smote)

def bar_as():
    oversample = SMOTE()
    X_train_res, y_train_res = oversample.fit_resample(train_sc, y_train)
    bar_after_smote = pd.value_counts(y_train_res)
    st.header("Label Sebelum SMOTE")
    st.bar_chart(bar_after_smote)


def before_smote():
    from collections import Counter
    counter = Counter(y_train)
    st.write("Jumlah ""%s""" % counter)
    # scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = np.where(y_train == label)[0]
        plt.scatter(X_train[row_ix, 0], X_train[row_ix, 1], label=str(label))
    plt.legend()
    st.pyplot(plt)

def after_smote():
    oversample = SMOTE()
    X_train_res, y_train_res = oversample.fit_resample(train_sc, y_train)
    from collections import Counter
    counter = Counter(y_train_res)
    st.write("Jumlah ""%s""" % counter)
    # scatter plot of examples by class label
    for label, _ in counter.items():
        # colors = np.random.rand(counter)
        row_ix = np.where(y_train_res == label)[0]
        plt.scatter(X_train_res[row_ix, 0], X_train_res[row_ix, 1], label=str(label))
        plt.legend()
    st.pyplot(plt)
# ----------------------------------- EXPLANATION -----------------------------------
#     in the end we come to the main function. which will return every value
#     from the above function. where the main function displays a summary page
#     of the code we have created. we will also call logo_(),
#     button_display() function (I think it will show on every page :D )
# ---------------------------------------------------------------------------------

if __name__ == '__main__':
    logo_()
    st.markdown("# Penentuan Klasifikasi UKT Berbasis "
                "*Machine Learning*")
                
    
    st.write("""Data Describe""")
    st.write(data.describe())

    # st.write("""%s""" % counter)

    kol1, kol2 = st.columns(2)

    with kol1:
        bar_bs()
        st.write("""# DataFrame Train""")
        st.bar_chart(pd.DataFrame(X_train))
        st.header("Sebelum SMOTE :")
        before_smote()

    with kol2:
        bar_as()
        # scatter_as()
        st.write("""# DataFrame Test""")
        st.bar_chart(pd.DataFrame(X_test))
        st.header("Sesudah SMOTE :")
        after_smote()

    pilih_model()
    # button_display()

    
    st.header('Hasil Smote :')
    smote()

    

