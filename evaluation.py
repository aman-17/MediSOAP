import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric
import torch

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['data'].tolist()

def generate_soap_note(model, tokenizer, conversation, max_length=512):
    inputs = tokenizer(conversation, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate(references, predictions):
    rouge = load_metric("rouge")
    bleu = load_metric("bleu")

    rouge_results = rouge.compute(predictions=predictions, references=references)
    bleu_results = bleu.compute(predictions=[[p.split()] for p in predictions], references=[[r.split()] for r in references])
    
    return rouge_results, bleu_results

def main(model_path, test_data_path, output_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    test_data = load_data(test_data_path)
    output_data = load_data(output_data_path)

    assert len(test_data) == len(output_data), "Mismatch between test data and output data lengths"

    references = []
    predictions = []
    
    for reference, conversation in zip(test_data, output_data):
        conversation, reference = conversation.split("</s>")[0].strip(), reference.split("</s>")[1].strip()
        prediction = generate_soap_note(model, tokenizer, conversation)
        predictions.append(prediction)
        references.append(reference)
    
    rouge_results, bleu_results = evaluate(references, predictions)

    print("ROUGE Results:", rouge_results)
    print("BLEU Results:", bleu_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned Llama2-7B model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to the test data CSV file")
    parser.add_argument("--output-data", type=str, required=True, help="Path to the output data CSV file")

    args = parser.parse_args()
    main(args.model_path, args.test_data, args.output_data)
