import json
from datasets import load_dataset
import random

dataset = load_dataset("xanderios/linkedin-job-postings", split="train")

def create_batch_file(job_postings, start_index, end_index, filename):
    with open(filename, "w") as f:
        for i, job in enumerate(job_postings[start_index:end_index], start=start_index):
            request = {
                "custom_id": f"request-{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are an AI assistant skilled in analyzing job postings and adding semantic tags to specific elements within the description."},
                        {"role": "user", "content": f"""Given the following job posting, wrap the relevant information with the specified tags where applicable. If any information is not present in the job posting, do not include the corresponding tag. Preserve the original text and structure of the job posting, only adding the tags where appropriate.

Tags to use:
<company_name></company_name>
<position></position>
<relevant_skills></relevant_skills>
<qualifications></qualifications>
<responsibilities></responsibilities>
<benefits></benefits>
<salary></salary>
<job_type></job_type>
<work_type></work_type>
<location></location>
<experience_level></experience_level>
<application_deadline></application_deadline>
<department></department>
<industry></industry>

Job Posting:
"{job['description']}"

Format your response as a JSON object with a single key "processed_description" containing the tagged job posting."""}
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}
                }
            }
            f.write(json.dumps(request) + "\n")
    print(f"Batch input file created: {filename}")

def get_custom_filename(batch_number):
    return f"job_posting_classification_batch_{batch_number}.jsonl"

num_job_postings = 33200
num_batches = 1

selected_indices = random.sample(range(len(dataset)), num_job_postings)
selected_job_postings = [dataset[i] for i in selected_indices]

items_per_batch = num_job_postings // num_batches

for i in range(num_batches):
    start_index = i * items_per_batch
    end_index = (i + 1) * items_per_batch if i < num_batches - 1 else num_job_postings
    filename = get_custom_filename(i + 1)
    create_batch_file(selected_job_postings, start_index, end_index, filename)

with open("linkedin_used_job_postings.json", "w") as f:
    json.dump({
        "job_posting_indices": selected_indices
    }, f)

print("All batch input files and used job postings file created successfully.")