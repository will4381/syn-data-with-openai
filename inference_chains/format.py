import json
import csv

def read_used_passages(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['passages']

def process_inferences_to_csv(inference_files, passages, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['passage', 'inference1', 'inference2', 'inference3', 'inference4', 'inference5']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file in inference_files:
            with open(file, 'r') as f:
                for line_number, line in enumerate(f, 1):
                    try:
                        response = json.loads(line)
                        if 'response' in response and response['response']['status_code'] == 200:
                            body = response['response']['body']
                            content = json.loads(body['choices'][0]['message']['content'])
                            inferences = content['inferences']

                            passage_index = int(response['custom_id'].split('-')[1]) - 1
                            passage = passages[passage_index]

                            row = {'passage': passage}
                            for i, inference in enumerate(inferences, 1):
                                row[f'inference{i}'] = inference

                            writer.writerow(row)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in file {file}, line {line_number}: {e}")
                        print(f"Problematic line: {line[:100]}...")
                    except Exception as e:
                        print(f"Error processing line {line_number} in file {file}: {e}")

    print(f"CSV file created: {output_file}")

if __name__ == "__main__":
    passages = read_used_passages("./data/msmarco_used_passages.json")

    inference_files = [
        "./data/combined_batch.jsonl"
    ]

    process_inferences_to_csv(inference_files, passages, "textual_inferences.csv")