"""Multi-label toxic comment classification with a Mixture-of-Experts BERT head."""

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

__version__ = "0.2.0"
__all__ = ["LABELS", "__version__"]
