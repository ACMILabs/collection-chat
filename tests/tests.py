import json
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.server import app, COLLECTION_API, ORGANISATION


def test_root():
    """
    Test the Collections chat root returns expected content.
    """
    client = TestClient(app)
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['message'] == f'Welcome to the {ORGANISATION} Collection Chat API.'
    assert f'{ORGANISATION} would like to acknowledge the Traditional Custodians'\
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
    assert f'{ORGANISATION} of Australia guide' in mock_llm.invoke.call_args[0][0]
    assert 'text=Oh+hello' in mock_llm.invoke.call_args[0][0]


@patch('api.server.llm')
@patch('api.server.docsearch')
def test_connection(mock_docsearch, mock_llm):
    """
    Test the /connection endpoint returns expected data.
    """
    mock_docsearch.get.return_value = {}
    mock_llm.invoke.return_value = MagicMock(content='An excellent connection.')

    client = TestClient(app)
    response = client.get('/connection')
    assert response.status_code == 422

    response = client.get('/connection?work_id=123')
    assert response.status_code == 404

    mock_docsearch.get.return_value = {
        'documents': [
            {'id': 123, 'title': 'Oh hello'},
            {'id': 456, 'title': 'Oh hi'},
        ],
        'embeddings': [[111, 222], [333, 444]],
    }
    response = client.get('/connection?work_id=123')
    assert response.status_code == 404

    mock_docsearch.similarity_search_by_vector.return_value = [
        MagicMock(page_content='Oh hello'),
        MagicMock(page_content='Oh hi'),
    ]
    response = client.get('/connection?work_id=123')
    assert response.status_code == 200
    mock_docsearch.get.assert_called_with(
        where={'source': f'{COLLECTION_API}123'},
        include=['embeddings', 'documents'],
    )
    mock_docsearch.similarity_search_by_vector.call_args.assert_called_with(
        [[111, 222], [333, 444]],
        k=4,
    )
    assert response.json()['work'] == '123'
    assert response.json()['works'][0]['text'] == 'Oh hello'
    assert response.json()['works'][1]['text'] == 'Oh hi'
    assert response.json()['connection'] == 'An excellent connection.'

    mock_docsearch.similarity_search_by_vector.return_value = [
        MagicMock(page_content=json.dumps({'id': 123, 'title': 'Oh hello'})),
        MagicMock(page_content=json.dumps({'id': 456, 'title': 'Oh hi'})),
    ]
    response = client.get('/connection?work_id=123')
    assert response.status_code == 200
    assert response.json()['work'] == '123'
    assert response.json()['works'][0]['id'] == 123
    assert response.json()['works'][1]['id'] == 456
    assert response.json()['connection'] == 'An excellent connection.'


@patch('api.server.requests.post')
def test_suggestions(mock_post):
    """
    Test the /suggestions endpoint returns the expected response.
    """
    client = TestClient(app)
    fake_response = MagicMock()
    fake_response.json.return_value = {'success': 'Suggestion created.'}
    fake_response.status_code = 200
    mock_post.return_value = fake_response
    sample_data = {
        'url': 'https://chat.acmi.net.au/?json=true&works=123',
        'text': 'A generated response.',
        'vote': 'up',
    }
    response = client.post('/suggestions', json=sample_data)

    assert response.status_code == 200
    assert response.json() == {'success': 'Suggestion created.'}

    mock_post.assert_called_once()
    _, called_kwargs = mock_post.call_args
    headers = called_kwargs.get('headers')
    data_passed = called_kwargs.get('data')
    assert headers.get('Content-Type') == 'application/json'
    assert 'Authorization' in headers
    assert data_passed == json.dumps(sample_data)


@patch('api.server.docsearch')
def test_works(mock_docsearch):
    """
    Test the /works endpoint returns expected data.
    """
    mock_docsearch.get.return_value = {}
    client = TestClient(app)
    response = client.get('/works')
    assert response.status_code == 404

    response = client.get('/works/123')
    assert response.status_code == 404

    mock_docsearch.get.return_value = {
        'documents': [
            '{"id": 123, "title": "Oh hello"}',
            '{"id": 456, "title": "Oh hi"}',
        ],
        'embeddings': [[111, 222], [333, 444]],
    }
    response = client.get('/works/123')
    assert response.status_code == 200
    mock_docsearch.get.assert_called_with(
        where={'source': f'{COLLECTION_API}123'},
        include=['embeddings', 'documents'],
    )
    assert response.json()['id'] == 123
