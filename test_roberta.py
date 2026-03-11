from brandprobe.scorers import SentimentWrapper

def test_roberta_sentiment():
    print("Testing RoBERTa Sentiment Wrapper...")
    
    positive_text = "I absolutely love this product! It's fantastic."
    negative_text = "I hate this. It is the worst thing ever, completely useless."
    neutral_text = "The package arrived yesterday."
    
    pos_score = SentimentWrapper.analyze(positive_text, method="roberta")
    neg_score = SentimentWrapper.analyze(negative_text, method="roberta")
    neu_score = SentimentWrapper.analyze(neutral_text, method="roberta")
    
    print(f"Positive score: {pos_score}")
    print(f"Negative score: {neg_score}")
    print(f"Neutral score: {neu_score}")

if __name__ == "__main__":
    test_roberta_sentiment()
