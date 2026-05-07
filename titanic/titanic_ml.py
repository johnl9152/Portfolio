import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("Target: Survived (0 = Did not survive, 1 = Survived)")
print("=" * 60)
print(f"\nTraining samples : {len(train_df)}")
print(f"Test samples     : {len(test_df)}")
print(f"\nSurvival rate    : {train_df['Survived'].mean():.1%}\n")

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()

    # Title from name
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare'
    )
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # Fill missing values
    df['Age'].fillna(df.groupby(['Title', 'Pclass'])['Age'].transform('median'), inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Age bands
    df['AgeBand'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100], labels=[0,1,2,3,4]).astype(int)

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    # Fare bands
    df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0,1,2,3]).astype(int)

    # Has cabin
    df['HasCabin'] = df['Cabin'].notna().astype(int)

    # Encode categoricals
    le = LabelEncoder()
    df['Sex']      = le.fit_transform(df['Sex'])          # male=1, female=0
    df['Embarked'] = le.fit_transform(df['Embarked'])
    df['Title']    = le.fit_transform(df['Title'])

    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

features = ['Pclass','Sex','AgeBand','FareBand','FamilySize','IsAlone',
            'HasCabin','Embarked','Title','SibSp','Parch']

X      = train_df[features]
y      = train_df['Survived']
X_test = test_df[features]

# ── Train & evaluate models ────────────────────────────────────────────────────
models = {
    'Logistic Regression' : LogisticRegression(max_iter=500, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=200, max_depth=6,
                                                    min_samples_split=4, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                        learning_rate=0.05, random_state=42),
    'SVM'                 : SVC(kernel='rbf', C=1, probability=True, random_state=42),
}

print("Cross-validation accuracy (5-fold):")
print("-" * 40)

best_score = 0
best_model = None
best_name  = ''

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean, std = scores.mean(), scores.std()
    print(f"  {name:<22} {mean:.4f} ± {std:.4f}")
    if mean > best_score:
        best_score = mean
        best_model = model
        best_name  = name

print(f"\nBest model: {best_name} ({best_score:.4f})\n")

# ── Detailed evaluation on hold-out set ───────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
val_preds = best_model.predict(X_val)

print("Hold-out validation results:")
print("-" * 40)
print(f"  Accuracy : {accuracy_score(y_val, val_preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, val_preds, target_names=['Did not survive (0)', 'Survived (1)']))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_val, val_preds)
print(f"  Not survived — TN: {cm[0,0]}  FP: {cm[0,1]}")
print(f"  Survived     — FN: {cm[1,0]}  TP: {cm[1,1]}")

# ── Feature importance (if available) ─────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importances:")
    print("-" * 40)
    importances = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = '█' * int(imp * 40)
        print(f"  {feat:<12} {imp:.4f}  {bar}")

# ── Final predictions on test set ─────────────────────────────────────────────
best_model.fit(X, y)  # retrain on full training data
test_preds = best_model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived'   : test_preds  # 0 or 1
})
submission.to_csv('submission.csv', index=False)

print(f"\nPrediction Summary:")
print("-" * 40)
print(f"  Total passengers predicted : {len(submission)}")
print(f"  Predicted to survive  (1)  : {(submission['Survived'] == 1).sum()}")
print(f"  Predicted not survive (0)  : {(submission['Survived'] == 0).sum()}")
print(f"  Predicted survival rate    : {submission['Survived'].mean():.1%}")
print("\nSubmission saved to: submission.csv")
print("=" * 60)
