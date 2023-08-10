import argparse
import json
from argparse import Namespace
from itertools import chain

import torch
from tqdm.auto import tqdm
from transformers import pipeline, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def generate(args, generator):
    output = generator("Hello, I'm a language model,", max_length=args.max_length, num_return_sequences=args.num_return_sequences)
    return output
    

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--max_length", type=int, default=100)
    parser.add_argument("-n" ,"--num_return_sequences", type=int, default=1)
    args = parser.parse_args()
    generator = pipeline('text-generation', model='gpt2', device=device)
    output = generate(args, generator)
    print(output)

def on_demand():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="outputs.json")
    parser.add_argument("-s", "--start", type=int, default=10)
    parser.add_argument("-e", "--end", type=int, default=300)
    parser.add_argument("-j", "--jump", type=int, default=10)
    parser.add_argument("-n" ,"--num_return_sequences", type=int, default=1)
    args = parser.parse_args()
    generator = pipeline('text-generation', model='gpt2', device=device)

    outputs = []
    for max_length in tqdm(range(args.start, args.end, args.jump)):
        temp_args = Namespace(max_length=max_length, num_return_sequences=args.num_return_sequences)
        output = generate(temp_args, generator)
        outputs.append(output)
    
    # merge all outputs
    merged_outputs = [x for x in chain.from_iterable([[p['generated_text'] for p in output] for output in outputs])]
    print(f"Total outputs: {len(merged_outputs)}")
    json.dump(merged_outputs, open(args.output, "w"), indent=4)
    

if __name__ == "__main__":
    on_demand()