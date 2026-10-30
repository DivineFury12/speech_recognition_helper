"""The API/model boundary must preserve request data and reject bad replies."""

import io
import unittest

import httpx
from fastapi.testclient import TestClient

from inference_app import create_inference_app
from src.adapters.inference_client import InferenceClient, RemoteEmbeddingModel
from src.errors import InvalidAudioError
from src.ports.embeddings import EmbeddingUnavailableError


class InferenceClientTests(unittest.TestCase):
    def client(self, handler):
        client = InferenceClient('http://inference:9000')
        client._http.close()
        client._http = httpx.Client(base_url='http://inference:9000',
                                   transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return client

    def test_audio_and_embedding_contract(self):
        seen = []

        def handle(request):
            seen.append(request)
            if request.url.path == '/health':
                return httpx.Response(200, json={'status': 'ok', 'model': 'gigaam', 'device': 'cpu'})
            if request.url.path == '/transcribe':
                return httpx.Response(200, json={'text': 'Сходить в магазин'})
            return httpx.Response(200, json={
                'model': 'intfloat/multilingual-e5-small', 'vector': [0.5] * 384,
            })

        client = self.client(handle)
        self.assertEqual(client.health()['device'], 'cpu')
        self.assertEqual(client.transribe(io.BytesIO(b'RIFFdata'), 'sample.wav'),
                         'Сходить в магазин')
        self.assertEqual(len(RemoteEmbeddingModel(client, 'e5').embed_query('магазин')), 384)
        self.assertIn(b'RIFFdata', seen[1].content)
        self.assertIn(b'"purpose":"query"', seen[2].content)

    def test_invalid_audio_and_wrong_vector(self):
        def handle(request):
            if request.url.path == '/transcribe':
                return httpx.Response(400, json={'detail': 'bad audio'})
            return httpx.Response(200, json={
                'model': 'wrong-model', 'vector': [1.0] * 384,
            })

        client = self.client(handle)
        with self.assertRaises(InvalidAudioError):
            client.transribe(io.BytesIO(b'invalid'))
        with self.assertRaises(EmbeddingUnavailableError):
            RemoteEmbeddingModel(client, 'e5').embed_document('текст')


class InferenceAppTests(unittest.TestCase):
    def test_internal_endpoints_delegate_to_model_ports(self):
        app = create_inference_app()

        class Transcriber:
            def transribe(self, source, filename):
                self.last = (source.read(), filename)
                return 'привет'

        class Embedding:
            name = 'intfloat/multilingual-e5-small'

            def embed_document(self, text):
                return [float(len(text))]

            def embed_query(self, text):
                return [float(len(text) + 1)]

        model = Transcriber()
        app.state.transcription = model
        app.state.embedding = Embedding()
        app.state.health = {'status': 'ok', 'model': 'gigaam', 'device': 'cpu'}
        client = TestClient(app)
        self.assertEqual(client.get('/health').json()['device'], 'cpu')
        self.assertEqual(client.post('/transcribe', files={'file': ('voice.wav', b'wave')}).json(),
                         {'text': 'привет'})
        self.assertEqual(model.last, (b'wave', 'voice.wav'))
        self.assertEqual(client.post('/embed', json={'text': 'дело', 'purpose': 'query'}).json(),
                         {'model': 'intfloat/multilingual-e5-small', 'vector': [5.0]})
        self.assertEqual(client.get('/speech/voices').json()['available'], False)


if __name__ == '__main__':
    unittest.main()
