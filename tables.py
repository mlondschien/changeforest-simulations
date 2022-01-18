import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from changeforest_simulations._load import load

MINIMAL_RELATIVE_SEGMENT_LENGTH = 0.01

# Table summarising datasets
for dataset in [
    "iris",
    "glass",
    "wine",
    "breast-cancer",
    "abalone",
    "dry-beans",
    "covertype",
]:
    data = load(dataset)

    small_classes = (
        data["class"]
        .value_counts(normalize=True)[lambda x: x < MINIMAL_RELATIVE_SEGMENT_LENGTH]
        .index
    )
    data = data[lambda x: ~x["class"].isin(small_classes)]

    X, y = data.drop(columns="class").to_numpy(), data["class"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    one_hot = np.apply_along_axis(lambda x: len(np.unique(x)) <= 2, 0, X).sum()
    _, value_counts = np.unique(y, return_counts=True)

    print(
        f"""\
{dataset.replace('-', ' ')} & \
{X.shape[0]} & {X.shape[1]} & \
{one_hot} & \
{len(np.unique(y))} & \
${', '.join([str(x) for x in sorted(value_counts)])}$ & \
{accuracy:.2f} \\\\"""
    )
