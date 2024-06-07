# https://github.com/krishnaik06/Finetuning-LLM/blob/main/Fine_tune_Llama_2.ipynb, https://github.com/believewhat/Dr.NoteAid, https://github.com/TheSeriousProgrammer/SimpleBitNet/tree/main, https://huggingface.co/datasets/omi-health/medical-dialogue-to-soap-summary
import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                # Extract the "dialogue" and "soap" fields
                dialogue = data.get("dialogue", "")
                soap = data.get("soap", "")
                new_data = {"dialogue": dialogue, "soap": soap}
                outfile.write(json.dumps(new_data) + '\n')
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")

input_file = 'data/validation.jsonl'
output_file = 'data/validation_output.jsonl'
process_jsonl(input_file, output_file)
