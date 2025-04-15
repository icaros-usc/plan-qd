from .huggingface import BaseLLM
import time
import openai
import asyncio
from typing import List

class SGLang(BaseLLM):
    """
    local llama 8b with SGLang cache reuse
    """
    def __init__(
        self,
        model_name: str, 
        key: str = None,
        base_url: str = "",
        max_tokens=1024,
        **kwargs,
    ):
        """NOTE: a server should be created before using sglang
        python -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-8B-Instruct --port 30000"""
        super().__init__(key, model_name, **kwargs)

        self.base_url = base_url

        # NOTE: our SGLang servers are hosted according to the following internal IP address
        self.async_client = openai.AsyncOpenAI(
            base_url=base_url, api_key="EMPTY"
        )

        self.client = openai.Client(
            base_url=base_url, api_key="EMPTY"
        )

    def query(self, messages, temp: int = 0.4):
        """
            SGLang backend query call, conform to OpenAI API format
        """
        response = self.client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=temp,
            max_tokens=1024,
            stream= False
        )

        return response.choices[0].message.content
    
    def batch_query_messages(
        self,
        all_messages: List,
        temp: int = 0.4,
        top_p: float = 0.95,
        top_k: int = 50
    ):
        """
        query uses OPENAI API format, input batch messages.
        Parameters:
            batch_messages (List[List[dict]]): batch of messages. 
        Returns:
            List[Dict]: List of responses in json format, one per message.
        """
        responses = []
        
        tries = 0
        
        while tries < 3:
            try:
                if len(all_messages) == 0:
                    return []

                async def gen_async(all_messages):
                    return await asyncio.gather(
                        *tuple(
                            self.async_client.chat.completions.create(
                                model="default",
                                messages=message,
                                temperature=temp,
                                max_tokens=1024,
                                stream=False,
                            )
                            for message in all_messages
                        )
                    )
                
                completions = asyncio.run(gen_async(all_messages))
                for completion in completions:
                    responses.append(completion.choices[0].message.content)
                
                break # break if this worked
            except Exception as e:
                print("Error in batch query", e)
                print("Sleeping for 30 seconds..")
                
                time.sleep(30)

                tries += 1
                if tries == 3:
                    raise e

        return responses

    def batch_query(
        self,
        context_batch,
        prompt_batch,
        temp: int = 0.4,
        top_p: float = 0.95,
        top_k: int = 50,
        chat=True
    ):
        all_messages = []

        for context, prompt in zip(context_batch, prompt_batch):
            if chat:
                templated_prompt = self.create_message_for_prompt(context, prompt)
            else:
                templated_prompt = f"{context}\n{prompt}"

            all_messages.append(templated_prompt)

        return self.batch_query_messages(all_messages, temp=temp, top_p=top_p, top_k=top_k)