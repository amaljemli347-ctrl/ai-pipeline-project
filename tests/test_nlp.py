import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nlp import TopicModeler, SentimentAndNERAnalyzer, evaluate_nlp_accuracy

def test_topic_modeler_short_doc_edge_case():
    tm = TopicModeler(n_topics=3)
    res = tm.predict_topic("Too short")
    assert res["topic_id"] == -1
    assert res["topic_name"] == "General/Short"

def test_topic_modeler_fitting_and_prediction():
    tm = TopicModeler(n_topics=3)
    docs = [
        "Computer science and software engineering are rapidly evolving fields.",
        "Artificial intelligence and deep learning models process large data.",
        "History and world wars shaped global governance and cultural heritage.",
        "Economic markets and financial systems experience inflation and growth."
    ]
    tm.fit(docs)
    res = tm.predict_topic("Software development with computer algorithms and code.")
    assert "topic_name" in res
    assert res["topic_id"] >= 0

def test_sentiment_analysis():
    analyzer = SentimentAndNERAnalyzer()
    pos_res = analyzer.analyze_sentiment("Excellent work with outstanding results and victory!")
    neg_res = analyzer.analyze_sentiment("Disastrous failure, loss, and war causing severe crisis.")
    neu_res = analyzer.analyze_sentiment("Standard file structure with configuration parameters.")
    
    assert pos_res["sentiment"] == "positive"
    assert neg_res["sentiment"] == "negative"
    assert neu_res["sentiment"] == "neutral"

def test_ner_extraction():
    analyzer = SentimentAndNERAnalyzer()
    text = "Microsoft was founded by Bill Gates in Redmond, Washington in 1975."
    res = analyzer.extract_entities(text)
    
    assert len(res["entities"]) > 0
    # Should identify ORG or PERSON or GPE or DATE
    entities_str = " ".join(res["entities"])
    assert "Bill Gates" in entities_str or "Microsoft" in entities_str or "1975" in entities_str or "Redmond" in entities_str

def test_evaluate_nlp_accuracy():
    eval_metrics = evaluate_nlp_accuracy()
    assert "accuracy" in eval_metrics
    assert eval_metrics["accuracy"] >= 0.7  # Expect at least 70%+ accuracy on manually labeled set
    assert eval_metrics["total_samples"] == 10
