import json
from datetime import datetime
from typing import List
import re

from llm_client import LLMClient
from tasks.base import Task
from utils import Example


def get_gold(example: dict) -> str:
    for answer in example["target_scores"]:
        if example["target_scores"][answer] == 1:
            return answer


class DateUnderstanding(Task):
    def __init__(self, llm: LLMClient):
        super().__init__("date", llm)

    def load_data(self) -> List[Example]:
        data = []
        with open("./data/date.json") as f:
            for example in json.load(f)["examples"]:
                data.append(Example(question=example["input"], answer=get_gold(example)))
        return data

    def extract_answer(self, raw_response: str) -> str:
        raw_response = raw_response.strip()

        # First try: search for a valid date format within the string using regex
        date_match = re.search(r'\d{2}/\d{2}/\d{4}', raw_response)
        if date_match:
            date_str = date_match.group()
            try:
                # Validate the date format
                datetime.strptime(date_str, "%m/%d/%Y")
                return date_str
            except ValueError:
                pass

        # Second try: extract answer using the "####" separator
        try:
            answer = raw_response.split("####")[1].strip()
            return self.extract_answer(answer)
        except Exception:
            pass

        print("Failed to extract answer from the following response:")
        print(raw_response)
        return "N/A"

    def equal(self, predicted_answer: str, expected_answer: str) -> bool:
        return predicted_answer == expected_answer

# date = DateUnderstanding("deepseek")
# response = "A: 04/01/2021"
# print(date.extract_answer(response))