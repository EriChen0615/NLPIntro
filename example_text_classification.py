from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("Vice President Kamala Harris went on the offensive on immigration at her rally in Atlanta Tuesday, attempting to counter former President Donald Trump’s attacks on the issue.")
print(result)