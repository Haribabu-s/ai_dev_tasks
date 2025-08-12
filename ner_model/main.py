import spacy
import json

# Load spaCy English model
# Run: python -m spacy download en_core_web_sm (once)
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extract named entities from the input text.
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })
    return entities

if __name__ == "__main__":
    # Read the accident report sample text
    with open("sample_data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Extract entities
    extracted_entities = extract_entities(text)

    # Print results in JSON format
    output = {
        "text": text,
        "entities": extracted_entities
    }
    print(json.dumps(output, indent=4))
