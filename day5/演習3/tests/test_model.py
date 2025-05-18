import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from exercise1.main import prepare_data, train_and_evaluate


@pytest.fixture(scope="module")
def model_and_data():
    """演習1の関数を使ってモデルとテストデータを取得"""
    X_train, X_test, y_train, y_test = prepare_data()
    model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
    return model, X_test, y_test

def test_model_accuracy(model_and_data):
    """モデルの精度が75%以上あることを検証"""
    model, X_test, y_test = model_and_data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.75, f"モデルの精度が低すぎます: {acc}"

def test_model_inference_time(model_and_data):
    """モデルの推論時間が1秒未満であることを検証"""
    model, X_test, _ = model_and_data
    start = time.time()
    model.predict(X_test)
    duration = time.time() - start
    assert duration < 1.0, f"推論が遅すぎます: {duration:.3f}秒"

def test_model_reproducibility():
    """同じ条件でモデルを2回作ったとき、結果が同じであることを検証"""
    X_train, X_test, y_train, y_test = prepare_data()
    model1, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
    model2, _ = train_and_evaluate(X_train, X_test, y_train, y_test)
    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)
    assert np.array_equal(pred1, pred2), "モデルの予測に再現性がありません"
