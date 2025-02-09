import re
from typing import Iterable
import requests
from bs4 import BeautifulSoup
import simplejson as json
from ollama import chat
from openai.types.chat import ChatCompletionMessageParam


class Website:

    headers = {
     "User-Agent": (
         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36")
    }

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if not soup.body:
            raise ValueError
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)


class Summarizer:
    model = "huihui_ai/deepseek-r1-abliterated:8b"
    agent_setup_messages: Iterable[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You are an assistant that analyzes the contents of a website " + 
                "and provides a short summary, ignoring text that might be navigation related. " +
                "Respond in markdown."
            )
        },
        {
            "role": "user",
            "content": (
                "My next message will be the content of the website that " +
                "I want you to make a short summary of as well as the title. " + 
                "If there are news or announcements, then summarize those " +
                "also. This message will be in json format like: " +
                "{'title': 'The title', 'content': 'parsed text content'}"
            )
        }
    ]

    def __init__(self, url):
        self.website = Website(url)
        self.messages = [
            *self.agent_setup_messages,
            {
                "role": "user",
                "content": json.dumps({
                    "title": self.website.title,
                    "content": self.website.text
                })
            }


        ]

    def strip_think(self, text: str):
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


    def summarize(self):
        response = chat(
            model = self.model,
            messages = self.messages
        )
        if not response.message.content:
            raise Exception("AI Response Error")
        return self.strip_think(response.message.content)


if __name__ == "__main__":
    print(Summarizer("https://www.yahoo.com/news/trump-netanyahu-hold-talks-us-050225710.html").summarize())
