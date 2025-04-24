import time
import csv
import os
from abc import ABC, abstractmethod
from typing import List, Literal

from tqdm import tqdm

from llm_client import LLMClient
from utils import Config, Example, compose_request, load_config


class Task(ABC):
    def __init__(self, name: str, llm: LLMClient, csv_file: str = "dataset.csv"):
        self.name = name
        self.llm = llm
        self.token_count_tracker = []
        self.latency_tracker = []
        self.csv_file = csv_file  # Path to the CSV file where data will be stored
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        # Check if the file exists and write the header if it's the first time
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['question', 'prompt_template', 'groq_output', 'token_count', 'is_correct'])

    @abstractmethod
    def load_data(self) -> List[Example]:
        pass

    @abstractmethod
    def extract_answer(self, raw_response: str) -> any:
        pass

    @abstractmethod
    def equal(self, predicted_answer: any, expected_answer: any) -> bool:
        pass

    def append_to_csv(self, question: str, prompt_template: str, groq_output: str, token_count: int, is_correct: bool):
        # Append data to CSV
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([question, prompt_template, groq_output, token_count, is_correct])

    def evaluate_example(self, model: str, config: Config, shot: int, example: Example, prompt_strategy: str) -> bool:
        # prepare payload
        payload = compose_request(config, shot, example.question)
        # print("composed request: ", payload)
        # run inference
        start_time = time.time()
        response, token_count = self.llm.request(payload)
        # print("response: ", response)
        # print("token count: ", token_count)
        end_time = time.time()
        self.token_count_tracker.append(token_count)
        self.latency_tracker.append(end_time - start_time)

        # check result
        predicted_answer = self.extract_answer(response)
        expected_answer = self.extract_answer(example.answer)
        equal = self.equal(predicted_answer, expected_answer)
        
        # Extract groq output (for simplicity, assuming the output format is expected)
        groq_output = response.strip()  # Adjust this if needed (for example, extracting final answer part)

        # Log the results to CSV
        self.append_to_csv(example.question, prompt_strategy, groq_output, token_count, equal)

        if not equal:
            print(f"Example: {example.question}")
            print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
            print(f"Full response: {response}")
        else:
            print("Example evaluated successfully!\n")
        return equal

    def evaluate(self, model: str, config: Literal["baseline", "cot", "cod"], shot: int = None) -> float:
        correct = 0
        prompt_strategy,config = load_config(self.name, config)
        test_set = self.load_data()
        for example in tqdm(test_set):
            if self.evaluate_example(model, config, shot, example,prompt_strategy):
                correct += 1
        return correct / len(test_set)
