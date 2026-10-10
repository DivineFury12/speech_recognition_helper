import types
import unittest

import torch

from src.adapters.local_embeddings import (
    E5EmbeddingModel, FridaEmbeddingModel, create_embedding_model,
)


class EmbeddingModelTests(unittest.TestCase):
    def test_factory_selects_compatible_models(self):
        self.assertIsInstance(create_embedding_model('e5', '/tmp/cache'), E5EmbeddingModel)
        self.assertIsInstance(create_embedding_model('frida', '/tmp/cache'), FridaEmbeddingModel)
        with self.assertRaises(ValueError):
            create_embedding_model('unknown', '/tmp/cache')

    def test_prefixes_and_dimensions(self):
        for model, expected_prefixes in (
            (E5EmbeddingModel('/tmp/cache'), ['query: запись', 'query: поиск']),
            (FridaEmbeddingModel('/tmp/cache'), [
                'search_document: запись', 'search_query: поиск',
            ]),
        ):
            with self.subTest(model=model.name):
                captured = []
                hidden = torch.zeros(1, 2, model.dimensions)
                hidden[0, 0, 0] = 1.0

                def tokenize(text, **kwargs):
                    captured.append(text)
                    return {'attention_mask': torch.tensor([[1, 0]])}

                model._tokenizer = tokenize
                model._model = lambda **kwargs: types.SimpleNamespace(last_hidden_state=hidden)
                self.assertEqual(len(model.embed_document('запись')), model.dimensions)
                self.assertEqual(len(model.embed_query('поиск')), model.dimensions)
                self.assertEqual(captured, expected_prefixes)


if __name__ == '__main__':
    unittest.main()
