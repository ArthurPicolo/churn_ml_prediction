# %%
# Importing Libs and Data
#===========================
import pandas as pd

from sklearn import tree
from sklearn import model_selection
from sklearn import ensemble
from sklearn import pipeline
from sklearn import metrics

import mlflow

from feature_engine import discretisation, encoding

import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id="102708712749657541")

pd.options.display.max_columns = 500
pd.options.display.max_rows = 500

df = pd.read_csv("data/abt_churn.csv")
df.head()

# %%
#Creating OOT sample for latter validation
oot = df[df["dtRef"] == df['dtRef'].max()].copy()
oot.head()

# %%
#Creating train sample
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train.shape

# %%
#Spliting the data between train and test
features = df_train.columns[2:-1]
target = 'flagChurn'

X, y = df_train[features], df_train[target]

# %%
# SAMPLE: Defing the rate
#===========================
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    random_state= 42,
                                                                    test_size=0.2,
                                                                    stratify= y
                                                                    )

print("Taxa Variável Resposta Treino: ", round(y_train.mean(),3))
print("Taxa Variável Resposta Teste: ", round(y_test.mean(),3))

# %%
# EDA
#===========================

#Explore Missing 
X_train.isna().sum().sort_values(ascending=False)


df_analise = X_train.copy()
df_analise[target] = y_train

sumario = df_analise.groupby(by=target).agg(["mean", "median"]).T

sumario['diff_abs'] = round(sumario[0] - sumario[1], 3)
sumario['diff_rel'] = round(sumario[0] / sumario[1], 3)

sumario.sort_values(by=['diff_rel'], ascending=False).head()

# %%
#Using tree to define the main features

# Main Features
arvore = tree.DecisionTreeClassifier(random_state=42, max_depth=5)
arvore.fit(X_train, y_train)

plt.figure(dpi=400, figsize=(4,4))
tree.plot_tree(arvore, feature_names=X_train.columns, 
               filled=True,
               class_names=[str(i) for i in arvore.classes_]
               )
plt.show()

#All Features
arvore_completa = tree.DecisionTreeClassifier(random_state=42)
arvore_completa.fit(X_train, y_train)

feature_importance = (pd.Series(arvore_completa.feature_importances_, index=X_train.columns)
                      .sort_values(ascending=False)
                      .reset_index()
                      )

feature_importance['acum.'] = feature_importance[0].cumsum()
feature_importance[feature_importance['acum.'] < 0.96]

best_features = (feature_importance[feature_importance['acum.'] < 0.96]['index'].tolist())

# %%
# MODIFY
#===========================

# Discretizer
tree_discretization = discretisation.DecisionTreeDiscretiser(
    variables=best_features,
    regression=False,
    bin_output='bin_number',
    cv=3,
)

# OneHot
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%
# MODEL
#===========================

with mlflow.start_run():

    mlflow.sklearn.autolog()

    model = ensemble.RandomForestClassifier(
        random_state=42,
        n_jobs=2,
    )

    run_name= model.__str__()

    params = {
        "min_samples_leaf":[15,20,25,30,50],
        "n_estimators":[100,200,500,1000],
        "criterion":['gini', 'entropy', 'log_loss'],
    }

    grid = model_selection.GridSearchCV(model,
                                        params,
                                        cv=3,
                                        scoring='roc_auc',
                                        verbose=4,
                                        )

    model_pipeline = pipeline.Pipeline(
        steps=[
            ('Discretizar', tree_discretization),
            ('Onehot', onehot),
            ('Grid',grid), 
        ]
    )

    model_pipeline.fit(X_train[best_features], y_train)

    ## ASSESS
    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)
    roc_train = metrics.roc_curve(y_train, y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)
    roc_test = metrics.roc_curve(y_test, y_test_proba)
    print("Acurácia Test:", acc_test)
    print("AUC Test:", auc_test)

    y_oot_predict = model_pipeline.predict(oot[best_features])
    y_oot_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_proba)
    roc_oot = metrics.roc_curve(oot[target], y_oot_proba)
    print("Acurácia oot:", acc_oot)
    print("AUC oot:", auc_oot)

    mlflow.log_metrics({
    "acc_train":acc_train,
    "auc_train":auc_train,
    "acc_test":acc_test,
    "auc_test":auc_test,
    "acc_oot":acc_oot,
    "auc_oot":auc_oot,
    })

# %%
# Ploting AUC Results
plt.figure(dpi=400)
plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title("Curva Roc")
plt.ylabel("Sensibilidade")
plt.xlabel("1- Especificidade")
plt.legend([
    f"Treino: {100*auc_train:.2f}%",
    f"Test: {100*auc_test:.2f}%",
    f"OOT: {100*auc_oot:.2f}%",
])
plt.show()
# %%
