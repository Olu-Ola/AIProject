import transformers
from transformers import AutoModelForSequenceClassification

import textattack
from textattack.models.helpers.lstm_for_classification import LSTMForClassification
from textattack.models.helpers.word_cnn_for_classification import WordCNNForClassification

class baseModel():

    def Bert(model_file):
        model = AutoModelForSequenceClassification.from_pretrained(model_file).to(textattack.shared.utils.device)
        tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
        model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
        return model_wrapper


    def LSTM(model_file):
        model = LSTMForClassification.from_pretrained(model_file).to(textattack.shared.utils.device)
        emb_layer = textattack.models.helpers.glove_embedding_layer.GloveEmbeddingLayer(emb_layer_trainable=False)
        word2id = emb_layer.word2id
        tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word2id, pad_token_id = 0, unk_token_id=100)
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)
        return model_wrapper


    def CNN(model_file):
        model = WordCNNForClassification.from_pretrained(model_file).to(textattack.shared.utils.device)
        emb_layer = textattack.models.helpers.glove_embedding_layer.GloveEmbeddingLayer(emb_layer_trainable=False)
        word2id = emb_layer.word2id
        tokenizer = textattack.models.tokenizers.glove_tokenizer.GloveTokenizer(word2id, pad_token_id = 0, unk_token_id=100)
        model_wrapper = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)
        return model_wrapper