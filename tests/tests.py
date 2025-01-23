import json
from unittest.mock import MagicMock, patch
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
    assert 'ACMI would like to acknowledge the Traditional Custodians'\
        in response.json()['acknowledgement']
    assert '/' in response.json()['api']
    assert '/docs' in response.json()['api']
    assert '/invoke' in response.json()['api']
    assert '/playground/{file_path:path}' in response.json()['api']
    assert '/similar' in response.json()['api']
    assert '/speak' in response.json()['api']
    assert '/summarise' in response.json()['api']

    response = client.get('/?json=false')
    assert response.status_code == 200
    assert 'The anecdote machine' in response.content.decode('utf-8')


@patch('api.server.Chroma.similarity_search_with_relevance_scores')
def test_similar(mock_similarity):
    """
    Test the /similar endpoint returns expected data.
    """
    mock_similarity.return_value = [
        (MagicMock(
            page_content=json.dumps({'key': 'value'}),
        ), 0.64),
    ]
    client = TestClient(app)
    response = client.get('/similar')
    assert response.status_code == 405

    response = client.post('/similar', data={'query': 'ghosts'})
    assert response.status_code == 200
    assert response.json()
    assert response.json()[0]['key'] == 'value'
    assert response.json()[0]['score'] == 0.64


@patch('api.server.ElevenLabs.generate')
def test_speak(mock_generate):
    """
    Test the /speak endpoint returns expected data.
    """
    mock_generate.return_value = iter([b'audio data'])

    client = TestClient(app)
    response = client.get('/speak')
    assert response.status_code == 405

    response = client.post('/speak', data={'text': 'Oh hello!'})
    assert response.status_code == 200
    assert response.content == b'audio data'


@patch('api.server.llm')
def test_summarise(mock_llm):
    """
    Test the /summarise endpoint returns expected data.
    """
    mock_llm.invoke.return_value = MagicMock(content='An excellent summary.')

    client = TestClient(app)
    response = client.get('/summarise')
    assert response.status_code == 405

    response = client.post('/summarise', data={'text': 'Oh hello!'})
    assert response.status_code == 200
    assert 'ACMI museum guide' in mock_llm.invoke.call_args[0][0]
    assert 'text=Oh+hello' in mock_llm.invoke.call_args[0][0]
