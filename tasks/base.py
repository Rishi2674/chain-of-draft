import time
from abc import ABC, abstractmethod
from typing import List, Literal
from conciseness_estimator.predict_strategy import predict_prompt_strategy
from tqdm import tqdm
from llm_client import LLMClient
from utils import Config, Example, compose_request, load_config


class Task(ABC):
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm
        self.token_count_tracker = []
        self.latency_tracker = []

    @abstractmethod
    def load_data(self) -> List[Example]:
        pass

    @abstractmethod
    def extract_answer(self, raw_response: str) -> any:
        pass

    @abstractmethod
    def equal(self, predicted_answer: any, expected_answer: any) -> bool:
        pass

    def evaluate_example(self, model: str, config: Config, shot: int, example: Example) -> bool:
        # prepare payload
        # if config.prompt_strategy == "pro_draft":
        #     prompt_strategy = predict_strategy(example.question)
        
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
        if not equal:
            print(f"Example: {example.question}")
            print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
            print(f"Full response: {response}")
        else:
            print("Example evaluated successfully!\n")
        return equal
    
    def evaluate_example_refined(self, model: str, config: Config, shot: int, example: Example ) -> bool:
        # prepare payload
        # if config.prompt_strategy == "pro_draft":
        #     prompt_strategy = predict_strategy(example.question)
        print("----------------------------------------")
        prompt_strategy = predict_prompt_strategy(example.question)
        print("Predicted Technique:",prompt_strategy)
        
        loaded_config = load_config(self.name, prompt_strategy)
        
        payload = compose_request(loaded_config, shot, example.question)
        
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
        if not equal:
            print(f"Example: {example.question}")
            print(f"Expected: {expected_answer}, Predicted: {predicted_answer}")
            print(f"Full response: {response}\n")
        else:
            print("Example evaluated successfully!\n")
        return equal

    def evaluate(self, model: str, config: Literal["baseline", "cot", "cod"], shot: int = None) -> float:
        correct = 0
        test_set = self.load_data()
        if config == "pro_draft":
            
            for example in tqdm(test_set):
                if self.evaluate_example_refined(model, config, shot, example):
                    correct += 1
            return correct / len(test_set)
        else: 
            config = load_config(self.name, config)
            for example in tqdm(test_set):
                if self.evaluate_example(model, config, shot, example):
                    correct += 1
            return correct / len(test_set)
