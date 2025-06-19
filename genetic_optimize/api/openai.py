from genetic_optimize.api.llmapi import LLMAPI

from openai import OpenAI

class OpenAIAPI(LLMAPI):
    def __init__(self, base_url, api_key, model="gpt-4o", test_connection=False):
        super().__init__(model)
        self.api_key = api_key
        self.base_url = base_url
        
        if test_connection:
            # 一个1x1像素的最小测试图片的base64编码
            test_img_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC"
            test_prompt = "This is a connection test."
            
            result = self.generate_text(test_prompt, test_img_base64)
            
            if result == "":
                raise ConnectionError(f"API连接测试失败。请检查您的API密钥、基础URL和网络连接。Model: {model}")
    
    def generate_text(self, prompt, imgbase64):
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        try:
            #提交信息至GPT4o
            response = client.chat.completions.create(
                model=self.model,#选择模型
                # model="deepseek-ai/Janus-Pro-7B",#选择模型
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional radiologist, specializing in interpreting medical images such as X-rays, CT scans, and MRIs. Please provide accurate assessments of the images based on the information provided, including possible diagnoses or recommendations for further tests. Ensure that your interpretations are based on standard medical knowledge and avoid making unverified claims."
                        # "content": "You are an expert medical imaging specialist."
                    },
                    {
                        "role": "user",
                        "content":[
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url":{
                                    "url": f"data:image/jpeg;base64,{imgbase64}"
                                }
                            },
                        ]
                    }
                ],
                stream=True,
            )

            reply = ""
            for res in response:
                if res.choices is None or len(res.choices) == 0:
                    continue
                if res.choices[0].delta is None:
                    continue
                content = res.choices[0].delta.content
                if content:
                    reply += content
        except Exception as e:
            print("Error: ", e)
            print("gpt failed")
            return ""
        return reply
