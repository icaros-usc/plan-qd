from .huggingface import BaseLLM

from openai import OpenAI

from logging import warning

from typing import List

class OpenAIClient(BaseLLM):
    def __init__(self, key, model_name):
        # create model using gpt-4
        super().__init__(key, model_name)

        client = OpenAI(api_key=key)
        self.client = client

    def query(self, messages, temp: int):
        request_reply = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,
            temperature=temp,
        )

        # return response only
        return request_reply.choices[0].message.content

    def batch_query(self, context_batch, prompt_batch, temp: int = 0.4, chat=True):
        if not chat:
            warning("OpenAI does not support non-chat completions.")

        responses = []
        for context, prompt in zip(context_batch, prompt_batch):
            messages = self.create_message_for_prompt(context, prompt)

            responses.append(
                self.query(messages, temp)
            )
        
        return responses
    
    def batch_query_messages(self, all_messages, temp: int = 0.4):
        responses = []

        for messages in all_messages:
            g = self.query(messages, temp)
            responses.append(g)
        
        
        return responses