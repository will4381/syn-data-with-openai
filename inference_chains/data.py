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

num_passages = 100000
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
                        {"role": "user", "content": f"""Given the following passage, generate a series of 3-5 logical inferences that can be drawn from the text. Ensure your inferences are well-reasoned, insightful, and directly related to the information provided in the passage. Consider both explicit and implicit information. Your inferences should demonstrate complex reasoning and may include deductions about causes, effects, motivations, or broader implications.

Passage: "{passage}"

Format your response as a JSON object with the following structure:

{{
    "inferences": [
        "First logical inference",
        "Second logical inference",
        "Third logical inference",
        "Fourth logical inference (if applicable)",
        "Fifth logical inference (if applicable)"
    ]
}}

Ensure each inference is distinct and adds new insight. Avoid repeating information directly stated in the passage."""}
                    ],
                    "max_tokens": 300,
                    "temperature": 0.7,
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"Batch input file created: {filename}")

def get_custom_filename(batch_number):
    return f"textual_inference_batch_{batch_number}.jsonl"

num_batches = 4
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