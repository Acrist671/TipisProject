import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MultiLabelBinarizer

@st.cache_data
def load_card_mapping():
    df = pd.read_csv("cardlist.csv")
    id_to_card = dict(zip(df['id'], df['card']))
    card_to_id = {card: id_ for id_, card in id_to_card.items()}
    return id_to_card, card_to_id

id_to_name, name_to_id = load_card_mapping()
ALL_CARD_NAMES = sorted(name_to_id.keys())
ALL_CARD_IDS = list(name_to_id.values())

@st.cache_resource
def load_model():
    return lgb.Booster(model_file='clash_royale_model.txt')

model = load_model()

mlb = MultiLabelBinarizer(classes=sorted(ALL_CARD_IDS))
mlb.fit([[]])

st.set_page_config(page_title="Clash Royale Predictor", layout="centered")
st.title("🎯 Предсказание победы в Clash Royale")
st.write("Выберите колоды игроков — модель оценит шанс победы Игрока 1.")

st.subheader("Колода Игрока 1")
deck1_names = []
for i in range(8):
    name = st.selectbox(
        f"Карта {i+1}",
        options=ALL_CARD_NAMES,
        key=f"p1_{i}"
    )
    deck1_names.append(name)

trophies_p1 = st.number_input("Трофеи Игрока 1", min_value=0, value=6000, step=100)

st.subheader("Колода Игрока 2")
deck2_names = []
for i in range(8):
    name = st.selectbox(
        f"Карта {i+1}",
        options=ALL_CARD_NAMES,
        key=f"p2_{i}"
    )
    deck2_names.append(name)

trophies_p2 = st.number_input("Трофеи Игрока 2", min_value=0, value=6000, step=100)

if st.button("Рассчитать шанс победы"):
    try:
        deck1_ids = [name_to_id[name] for name in deck1_names]
        deck2_ids = [name_to_id[name] for name in deck2_names]

        d1 = mlb.transform([deck1_ids])
        d2 = mlb.transform([deck2_ids])

        trophy_diff = trophies_p1 - trophies_p2
        features = np.concatenate([d1[0], d2[0], [trophies_p1, trophies_p2, trophy_diff]]).reshape(1, -1)

        proba = model.predict(features)[0]
        win_chance = proba * 100

        st.success(f"🏆 Шанс победы Игрока 1: **{win_chance:.2f}%**")
        st.progress(min(int(win_chance), 100))

    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.exception(e)