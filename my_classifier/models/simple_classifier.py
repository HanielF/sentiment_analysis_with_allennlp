from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
import sys
import logging
from allennlp.nn.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator

logger = logging.getLogger(__name__)


@Model.register("bert_pool_classifier")
class BertPoolClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        #  logger.info("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        logger.info("==> text input: {} ".format(text))
        embedded_text = self.embedder(text)
        logger.info("==> embedding shape: {} ".format(embedded_text.shape))
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        logger.info("==> encoded shape: {}".format(encoded_text.shape))
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        logger.info("==> logits shape: {}".format(logits.shape))
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        logger.info("==> softmax probs shape: {}".format(probs.shape))
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register("lstm_classifier")
class LSTMClassifier(Model):
    def __init__(
            self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, feedforward: FeedForward):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        print("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        #  logger.info("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        self.feedforward = feedforward
        self.classifier = torch.nn.Linear(self.feedforward.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        #  print("==> encoded_text shape: {}\n".format(encoded_text.shape))
        feedforward_text = self.feedforward(encoded_text)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(feedforward_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

@Model.register("base_lstm_classifier")
class BaseLSTMClassifier(Model):
    def __init__(
            self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, feedforward: FeedForward):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        print("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        #  logger.info("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        self.feedforward = feedforward
        self.classifier = torch.nn.Linear(self.feedforward.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        #  print("==> encoded_text shape: {}\n".format(encoded_text.shape))
        feedforward_text = self.feedforward(encoded_text)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(feedforward_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



@Model.register("roberta_pool_classifier")
class RobertaPoolClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        print("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        logger.info("==> encoded input shape: {}, output shape: {}\n".format(encoder.get_input_dim(),encoder.get_output_dim()))
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        #  logger.info("==> embedding shape: ".format(embedded_text.shape))
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        #  logger.info("==> encoded shape: ".format(type(encoded_text)))
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        #  logger.info("==> logits shape: ".format(logits.shape))
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        #  logger.info("==> probs shape: ".format(probs.shape))
        #  Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


@Model.register("text_cnn_classifier")
class TextCnnClassifier(Model):
    def __init__(self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder, dropout: float, regularizer: RegularizerApplicator=None) -> None:
        super().__init__(vocab, regularizer)
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = torch.nn.Linear(self.encoder.get_output_dim(), vocab.get_vocab_size("labels"))
        self.accuracy = CategoricalAccuracy()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text: TextFieldTensors, label: torch.Tensor=None)->Dict[str, torch.Tensor]:
        embedding = self.embedder(text)
        mask = util.get_text_field_mask(text)

        encoded_text = self.encoder(embedding, mask)
        dropout_encode = self.dropout(encoded_text)

        logits = self.classifier(dropout_encode)
        probs = torch.nn.functional.softmax(logits)
        output = {"probs":probs}

        if label is not None:
            self.accuracy(logits, label)
            output['loss']=torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset:bool=False)->Dict[str, float]:
        return {"accuracy":self.accuracy.get_metric(reset)}

