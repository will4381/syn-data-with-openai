import json
import csv

def read_used_job_postings(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['job_posting_indices']

def process_job_postings_to_csv(batch_files, dataset, used_indices, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['original_description', 'processed_description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in batch_files:
            with open(file, 'r') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        response = json.loads(line)
                        if 'response' in response and response['response']['status_code'] == 200:
                            body = response['response']['body']
                            content = json.loads(body['choices'][0]['message']['content'])
                            processed_description = content['processed_description']
                            
                            job_posting_index = int(response['custom_id'].split('-')[1]) - 1
                            original_description = dataset[used_indices[job_posting_index]]['description']
                            
                            row = {
                                'original_description': original_description,
                                'processed_description': processed_description
                            }
                            writer.writerow(row)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in file {file}, line {line_number}: {e}")
                        print(f"Problematic line: {line[:100]}...")
                    except Exception as e:
                        print(f"Error processing line {line_number} in file {file}: {e}")

    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("xanderios/linkedin-job-postings", split="train")

    used_indices = read_used_job_postings("./data/linkedin_used_job_postings.json")

    batch_files = [
        "./data/job_tagged.jsonl"
    ]

    process_job_postings_to_csv(batch_files, dataset, used_indices, "job_posting_classifications.csv")