from unittest.mock import MagicMock

from llm_mitm.auth import get_key_id


def _request(headers: dict) -> MagicMock:
    req = MagicMock()
    req.headers = headers
    return req


def test_hashes_bearer_token_by_default():
    req = _request({"authorization": "Bearer sk-my-secret-key"})
    key_id = get_key_id(req)
    assert key_id != "sk-my-secret-key"
    assert len(key_id) == 16


def test_hashing_is_deterministic():
    req = _request({"authorization": "Bearer sk-my-secret-key"})
    assert get_key_id(req) == get_key_id(req)


def test_different_keys_hash_differently():
    req1 = _request({"authorization": "Bearer key-one"})
    req2 = _request({"authorization": "Bearer key-two"})
    assert get_key_id(req1) != get_key_id(req2)


def test_hash_api_keys_false_returns_raw_key():
    req = _request({"authorization": "Bearer sk-my-secret-key"})
    assert get_key_id(req, hash_api_keys=False) == "sk-my-secret-key"


def test_bearer_prefix_is_case_insensitive():
    req = _request({"authorization": "bearer sk-my-secret-key"})
    assert get_key_id(req, hash_api_keys=False) == "sk-my-secret-key"


def test_falls_back_to_x_api_key_header():
    req = _request({"x-api-key": "sk-other-key"})
    assert get_key_id(req, hash_api_keys=False) == "sk-other-key"


def test_falls_back_to_api_key_header():
    req = _request({"api-key": "sk-other-key"})
    assert get_key_id(req, hash_api_keys=False) == "sk-other-key"


def test_anonymous_when_no_key_present():
    req = _request({})
    assert get_key_id(req) == "anonymous"
