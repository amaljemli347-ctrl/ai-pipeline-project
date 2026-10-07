import re
import logging
from typing import List, Dict, Any, Tuple
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger("NLP_Analysis")

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logger.warning("en_core_web_sm not found, attempting to load blank 'en'")
    nlp = spacy.blank("en")

# Define fallback topic names for LDA topic indices
TOPIC_NAMES = {
    0: "Science & Technology",
    1: "History & Culture",
    2: "Business & Economy",
    3: "Society & Governance",
    4: "Arts & Media"
}

# Positive / Negative lexicon for rule-based sentiment scoring
POSITIVE_WORDS = {"great", "good", "excellent", "positive", "success", "innovative", "leading", "popular", "famous", "hero", "victory", "growth", "achievement", "benefit", "best", "advancement"}
NEGATIVE_WORDS = {"bad", "poor", "failure", "negative", "loss", "war", "crisis", "conflict", "death", "error", "destroy", "problem", "disaster", "harm", "worst", "decline"}

class TopicModeler:
    """
    Topic Modeling engine using Latent Dirichlet Allocation (LDA).
    Handles edge cases like short documents or jargon.
    """
    def __init__(self, n_topics: int = 5, max_features: int = 1000):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
        self.lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.is_fitted = False

    def fit(self, documents: List[str]):
        """Fits the LDA model on the corpus."""
        valid_docs = [d for d in documents if len(d.strip()) > 20]
        if not valid_docs:
            logger.warning("Not enough valid documents to fit topic model.")
            return
        
        tf = self.vectorizer.fit_transform(valid_docs)
        self.lda.fit(tf)
        self.is_fitted = True
        logger.info(f"TopicModeler successfully fitted with {self.n_topics} topics.")

    def predict_topic(self, text: str) -> Dict[str, Any]:
        """
        Predicts dominant topic for a single document.
        Handles short document edge case.
        """
        if not text or len(text.strip()) < 20:
            return {
                "topic_id": -1,
                "topic_name": "General/Short",
                "confidence": 1.0
            }
            
        if not self.is_fitted:
            return {
                "topic_id": 0,
                "topic_name": TOPIC_NAMES[0],
                "confidence": 0.5
            }
            
        try:
            tf = self.vectorizer.transform([text])
            probs = self.lda.transform(tf)[0]
            top_topic_id = int(probs.argmax())
            confidence = float(probs[top_topic_id])
            
            return {
                "topic_id": top_topic_id,
                "topic_name": TOPIC_NAMES.get(top_topic_id, f"Topic_{top_topic_id}"),
                "confidence": round(confidence, 3)
            }
        except Exception as e:
            logger.error(f"Error predicting topic: {e}")
            return {"topic_id": -1, "topic_name": "General/Unknown", "confidence": 0.0}


class SentimentAndNERAnalyzer:
    """
    Extracts Named Entities (PERSON, ORG, GPE, DATE) and computes sentiment polarity.
    """
    def __init__(self):
        self.nlp = nlp

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Computes sentiment polarity (positive, negative, neutral) using lexicon analysis.
        """
        if not text or len(text.strip()) == 0:
            return {"sentiment": "neutral", "score": 0.0}
            
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return {"sentiment": "neutral", "score": 0.0}
            
        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        
        score = (pos_count - neg_count) / max(len(words), 1) * 10
        
        if score > 0.1 or pos_count > neg_count:
            sentiment = "positive"
        elif score < -0.1 or neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "sentiment": sentiment,
            "score": round(score, 3)
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts Named Entities using spaCy.
        """
        if not text or len(text.strip()) == 0:
            return {"entities": []}
            
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "EVENT", "NORP"}:
                entities.append(f"{ent.text} ({ent.label_})")
                
        return {
            "entities": list(set(entities))[:10]  # Cap at 10 distinct entities
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Runs both sentiment and NER analysis."""
        sentiment_res = self.analyze_sentiment(text)
        ner_res = self.extract_entities(text)
        return {
            "sentiment": sentiment_res["sentiment"],
            "sentiment_score": sentiment_res["score"],
            "entities": ", ".join(ner_res["entities"]) if ner_res["entities"] else "None"
        }


# --- EVALUATION DATASET & METRICS FOR WEEK 4 ---
MANUAL_TEST_SET = [
    {"text": "Apple Inc. achieved record revenue growth and great customer satisfaction in 2024.", "expected_sentiment": "positive"},
    {"text": "The company suffered a severe financial crisis and catastrophic loss during the recession.", "expected_sentiment": "negative"},
    {"text": "The document contains standard reference technical parameters for system configuration.", "expected_sentiment": "neutral"},
    {"text": "Albert Einstein published the general theory of relativity in Berlin, Germany.", "expected_sentiment": "positive"},
    {"text": "Disastrous conflict and destruction engulfed the region following the war.", "expected_sentiment": "negative"},
    {"text": "Python is an interpreted high-level general-purpose programming language.", "expected_sentiment": "neutral"},
    {"text": "The team celebrated a monumental victory and impressive achievement.", "expected_sentiment": "positive"},
    {"text": "System failure occurred due to a critical memory corruption error.", "expected_sentiment": "negative"},
    {"text": "The meeting was held on Monday morning in London with board members.", "expected_sentiment": "neutral"},
    {"text": "Innovative medical research brought major health benefits and success to patients.", "expected_sentiment": "positive"},
]

def evaluate_nlp_accuracy() -> Dict[str, Any]:
    """
    Evaluates Sentiment Analysis against the manually labeled test dataset.
    Computes Accuracy, Precision, Recall, and F1 Score.
    """
    analyzer = SentimentAndNERAnalyzer()
    
    correct = 0
    total = len(MANUAL_TEST_SET)
    
    y_true = []
    y_pred = []
    
    for item in MANUAL_TEST_SET:
        res = analyzer.analyze_sentiment(item["text"])
        pred = res["sentiment"]
        gold = item["expected_sentiment"]
        
        y_true.append(gold)
        y_pred.append(pred)
        
        if pred == gold:
            correct += 1
            
    accuracy = correct / total
    
    # Calculate macro precision, recall, f1 across classes (positive, negative, neutral)
    classes = ["positive", "negative", "neutral"]
    f1_scores = []
    
    for c in classes:
        tp = sum(1 for gt, p in zip(y_true, y_pred) if gt == c and p == c)
        fp = sum(1 for gt, p in zip(y_true, y_pred) if gt != c and p == c)
        fn = sum(1 for gt, p in zip(y_true, y_pred) if gt == c and p != c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)
        
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    return {
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "total_samples": total,
        "correct_predictions": correct
    }
