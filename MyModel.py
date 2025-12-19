import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

df = pd.read_csv("data_ord.csv")

df = df.iloc[:, 1:]

cards = df.iloc[:, :16].values  
trophies_p1 = df.iloc[:, 16].values   
trophies_p2 = df.iloc[:, 17].values
trophy_diff = (trophies_p1 - trophies_p2).reshape(-1, 1)
y = df.iloc[:, -1].astype(int).values 

deck1 = cards[:, :8]
deck2 = cards[:, 8:]

mlb = MultiLabelBinarizer(classes=range(0, 106))

deck1_ohe = mlb.fit_transform(deck1)
deck2_ohe = mlb.fit_transform(deck2)

print("Форма one-hot колод:", deck1_ohe.shape)

X = np.concatenate([deck1_ohe, deck2_ohe,
                    trophies_p1.reshape(-1,1),
                    trophies_p2.reshape(-1,1),
                    trophy_diff], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data],
    num_boost_round=2000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)

y_proba = model.predict(X_test)
print(f"ROC-AUC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"Log-loss:    {log_loss(y_test, y_proba):.4f}")

model.save_model('clash_royale_model.txt')
print("✅ Модель сохранена!")
