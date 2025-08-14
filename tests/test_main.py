import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import predict_cp


def test_predict_alpha():
    result = predict_cp("alpha colony 123456")
    assert result['predicted_cp'] == 'CP_A'
    assert 0.0 <= result['confidence'] <= 1.0


def test_predict_beta():
    result = predict_cp("beta colony 123456")
    assert result['predicted_cp'] == 'CP_B'
    assert 0.0 <= result['confidence'] <= 1.0
