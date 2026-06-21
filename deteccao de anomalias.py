import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


url="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df=pd.read_csv(url)
df.head()

df ['Class'].value_counts(normalize = True)

df ['amount_log'] = np.log1p(df['Amount'])

scaler = StandardScaler ()
df ["amount_scaled"] = scaler.fit_transform(df[["amount_log"]])

X = df.drop("Class", axis = 1)
y = df ["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = LogisticRegression(max_iter = 5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


y_probs = model.predict_proba(X_test)
y_probs = y_probs[:, 1]
fpr, tpr,_= roc_curve(y_test, y_probs)

plt.plot (fpr,tpr)
plt.title ("ROC curve")
plt.xlabel ("False Positive Rate")
plt.ylabel ("True Positive Rate")
plt.show()

#undersampling
fraudes = df[df["Class"] == 1]
nao_fraudes = df[df["Class"] == 0].sample(len(fraudes),random_state=42)

df_under = pd.concat([fraudes, nao_fraudes])

#oversambpling
smote = SMOTE()
x_res,y_res = smote.fit_resample(X,y)

rf = RandomForestClassifier(
    n_estimator = 50, max_depth = 10,
    class_weight = "balanced",
    n_jobs = -1,
    random_state = 42
)
rf.fit(x_res,y_res)
y_pred_ref = rf.predict(X_test)

pipeline = Pipeline ([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter = 5000))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

tresh = 0.3
y_pred_custom = (y_probs > tresh).astype(int)

xgb = XGBClassifier(
    scale_pos_weight = 10,
    use_label_encoder = False,
    eval_metric = "logloss",
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

importancias = xgb.feature_importances()
plt.bar (range(len (importancias )), importancias)
plt.title("Importancia das Variavéis")
plt.show()

param_grid = {
    "max_depth": [3,5],
    "n_estimators": [50,100],
}

grid = GridSearchCV(
    XGBClassifier(eval_metric="logloss"),
    param_grid,
    scoring = "recall",
    cv = 3
)
grid.fit(X_train, y_train)
y_pred_grid = grid.predict(X_test)


print (classification_report(y_test, y_pred))
print ("AUC: ", roc_auc_score(y_test, y_probs))
print (classification_report(y_test, y_pred))
print (classification_report (y_test,y_pred_custom))
print ("Melhores Modelos", grid.best_params_)

#

