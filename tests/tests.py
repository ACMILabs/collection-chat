from fastapi.testclient import TestClient

from api.server import app


def test_root():
    """
    Test the Collections chat root returns expected content.
    """
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['message'] == 'Welcome to the ACMI Collection Chat API.'
    assert '/' in response.json()['api']
    assert '/docs' in response.json()['api']
    assert '/invoke' in response.json()['api']
    assert '/playground/{file_path:path}' in response.json()['api']