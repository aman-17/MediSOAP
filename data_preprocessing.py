import json
import csv

class DataTransformer:
    def __init__(self, jsonl_file, output_csv_file):
        self.jsonl_file = jsonl_file
        self.output_csv_file = output_csv_file

    def llama_template(self):
        transformed_data = []
        with open(self.jsonl_file, 'r') as infile:
            for line in infile:
                try:
                    example = json.loads(line)
                    dialogue = example.get("dialogue", "").replace('\n', ' ').strip()
                    soap = example.get("soap", "").replace('\n', ' ').strip()
                    # Apply the Llama2 template
                    transformed_text = f'<s>[INST] {dialogue} [/INST] {soap} </s>'
                    transformed_data.append({"data": transformed_text})
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
        
        return transformed_data

    def save_to_csv(self, data):
        with open(self.output_csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['data'])
            writer.writeheader()
            writer.writerows(data)

    def process(self):
        transformed_data = self.llama_template()
        self.save_to_csv(transformed_data)
        print(f"Processed data has been saved to {self.output_csv_file}")

files = [
    {'jsonl_file': 'data/train.jsonl', 'output_csv_file': 'data/train_llama_formatted.csv'},
    {'jsonl_file': 'data/validation.jsonl', 'output_csv_file': 'data/validation_llama_formatted.csv'},
    {'jsonl_file': 'data/test.jsonl', 'output_csv_file': 'data/test_llama_formatted.csv'}
]

for file in files:
    transformer = DataTransformer(file['jsonl_file'], file['output_csv_file'])
    transformer.process()

