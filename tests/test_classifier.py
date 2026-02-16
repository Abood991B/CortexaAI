from agents.classifier import DomainClassifier

classifier = DomainClassifier()

# Test the classifier with a simple prompt
test_prompt = "Write a function to sort a list of numbers"

try:
    result = classifier.classify_prompt(test_prompt)
    print("Classification successful:")
    print(result)
except Exception as e:
    print(f"Error: {e}")
