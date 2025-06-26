# train_models.py
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from ai_detector import vectorize_input

# 데이터 로드 (CSV 파일 경로 설정)
DATA_PATH = os.path.join(os.getcwd(), 'data', 'sqli_dataset.csv')
data = pd.read_csv(DATA_PATH)

# 피처/레이블 준비
X = data.apply(lambda r: vectorize_input(r['url'], r['param'], r['value']), axis=1).tolist()
y = data['label'].tolist()

# 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 검증 및 성능 출력
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:,1]
print(classification_report(y_val, y_pred))
print('ROC AUC:', roc_auc_score(y_val, y_proba))

# 모델 저장 (models 디렉터리에 자동 생성)
os.makedirs('models', exist_ok=True)
joblib.dump(model, os.path.join('models', 'sqli_detector.pkl'))