import json
from datasets import load_dataset
import random

dataset = load_dataset("Tevatron/msmarco-passage-corpus", split="train")

def select_passages(dataset, num_passages):
    selected_passages = []
    passage_ids = []
    
    length_categories = [
        (0, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300)
    ]
    
    passages_per_category = num_passages // len(length_categories)
    
    for start, end in length_categories:
        category_passages = [
            (i, p['text']) for i, p in enumerate(dataset) 
            if start <= len(p['text'].split()) <= end
        ]
        selected = random.sample(category_passages, min(passages_per_category, len(category_passages)))
        selected_passages.extend([p for _, p in selected])
        passage_ids.extend([i for i, _ in selected])
    
    while len(selected_passages) < num_passages:
        i = random.randint(0, len(dataset) - 1)
        if i not in passage_ids:
            selected_passages.append(dataset[i]['text'])
            passage_ids.append(i)
    
    return selected_passages, passage_ids

num_passages = 50000
selected_passages, passage_ids = select_passages(dataset, num_passages)

def create_batch_file(passages, start_index, end_index, filename):
    with open(filename, "w") as f:
        for i, passage in enumerate(passages[start_index:end_index], start=start_index):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are an AI assistant skilled in logical reasoning and drawing inferences from text."},
                        {"role": "user", "content": f"""Given the following passage, generate a series of 5 inferences that can be drawn from the text. Include a mix of well-reasoned, insightful inferences as well as some that may be less supported or even incorrect. Assign each inference a confidence score between 0 and 1, where 1 indicates high confidence in the inference's correctness and relevance, and 0 indicates low confidence.

Passage: "{passage}"

Format your response as a JSON object with the following structure:
{{
 "inferences": [
   {{
     "text": "First inference",
     "confidence": 0.XX
   }},
   {{
     "text": "Second inference",
     "confidence": 0.XX
   }},
   {{
     "text": "Third inference",
     "confidence": 0.XX
   }},
   {{
     "text": "Fourth inference",
     "confidence": 0.XX
   }},
   {{
     "text": "Fifth inference",
     "confidence": 0.XX
   }}
 ]
}}

Ensure each inference is distinct and varies in its level of correctness and relevance to the passage. Include:
1. At least one highly plausible and well-supported inference (confidence > 0.80)
2. At least one inference that's somewhat plausible but not strongly supported (confidence 0.40 - 0.60)
3. At least one inference that's questionable or a stretch based on the given information (confidence < 0.30)

The other inferences can fall anywhere on this spectrum. Avoid repeating information directly stated in the passage."""}
                    ],
                    "max_tokens": 2048,
                    "temperature": 1.0,
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"Batch input file created: {filename}")

def get_custom_filename(batch_number):
    return f"textual_inference_batch_{batch_number}.jsonl"

num_batches = 1
items_per_batch = num_passages // num_batches

for i in range(num_batches):
    start_index = i * items_per_batch
    end_index = (i + 1) * items_per_batch if i < num_batches - 1 else num_passages
    filename = get_custom_filename(i + 1)
    create_batch_file(selected_passages, start_index, end_index, filename)

with open("msmarco_used_passages.json", "w") as f:
    json.dump({
        "passage_ids": passage_ids,
        "passages": selected_passages
    }, f)

print("All batch input files and used passages file created successfully.")