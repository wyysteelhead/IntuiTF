import json
import requests
from genetic_optimize.api.llmapi import LLMAPI


class GeminiAPI(LLMAPI):
    def __init__(self, base_url, api_key, model="gemini-2.0-flash-001"):
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url
    
    def generate_text(self, prompt, imgbase64):
        URL = f"{self.base_url}/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        print(">DEBUG: API URL:", URL)
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": imgbase64
                        }
                    }
                ]
            }]
        }

        # Send request
        headers = {"Content-Type": "application/json"}
        response = requests.post(URL, headers=headers, data=json.dumps(payload))
        response = response.json()
        # Return result
        return response['candidates'][0]['content']['parts'][0]['text']