import random
import cv2
import numpy as np
import openai
import base64
from openai import OpenAI
from PIL import Image, ImageDraw
import os
import json
from io import BytesIO
import uuid
from datetime import datetime
from genetic_optimize.api.gemini import GeminiAPI
from genetic_optimize.api.openai import OpenAIAPI
from genetic_optimize.states.direction import Direction
from genetic_optimize.utils.image_utils import *
from genetic_optimize.utils.file_utils import *
from genetic_optimize.states.evaluation_state import *

class LLM_Evaluator:
    def __init__(self, base_url, api_key, prompt_folder, quality_metrics = None, text_metrics = None, instruct_number=None, model_name="gpt-4o"):
        self.api_key=api_key
        self.base_url=base_url
        self.prompt_folder = prompt_folder
        self.prompt_format = load_prompt(os.path.join(prompt_folder, 'prompt_format.txt'))
        self.prompt_format_gaussian = load_prompt(os.path.join(prompt_folder, 'prompt_format_gaussian.txt'))
        self.aspects, self.modes = load_aspects(os.path.join(prompt_folder, 'aspects.json'), quality_metrics=quality_metrics, text_metrics=text_metrics)
        instruct_path = os.path.join(prompt_folder, 'instructions.json')
        self.instruct = load_instruct(instruct_path=instruct_path, instruct_number=instruct_number)
        self.middle_img = None
        self.modification = None
        print("using model:", model_name)
        # if model_name contains "gpt":
        if "gpt" in model_name.lower():
            self.api = OpenAIAPI(base_url=base_url, api_key=api_key, model=model_name)
        elif "gemini" in model_name.lower():
            # self.api = GeminiAPI(base_url=base_url, api_key=api_key, model=model_name)
            self.api = OpenAIAPI(base_url=base_url, api_key=api_key, model=model_name)
        else:
            raise ValueError(f"Model not supported: {model_name}.")
        
    @staticmethod
    def get_all_image_paths(folder_path):
        """
        从指定文件夹中读取所有图片的路径。
        """
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        return [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if os.path.isfile(os.path.join(folder_path, f)) and os.path.splitext(f)[1].lower() in supported_extensions
        ]

    @staticmethod
    def process_image_with_gpt4o(client, base64_image, prompt):
        """
        将内存中的图片传递给模型，并返回输出。
        """
        # base64_image = base64.b64encode(img_base64.read()).decode('utf-8')
        try:
            #提交信息至GPT4o
            response = client.chat.completions.create(
                model="gpt-4o",#选择模型
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
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ]
                    }
                ],
                stream=True,
            )

            reply = ""
            for res in response:
                if res.choices[0].delta is None:
                    continue
                content = res.choices[0].delta.content
                if content:
                    reply += content
        except Exception:
            # print the error message
            print("Error: ", Exception)
            print("gpt failed")
            return ""
        return reply
        
    @staticmethod
    def process_mutate_output(gpt_output):
        if "action:" not in gpt_output.lower():
            return -2, None
        
        action_line = gpt_output.lower().split("action:")[1].strip()
        # the final answer line is an action to move the gaussian or recolor the gaussian, the actions are formulated in json style, for example: {"color": {"hues": 0.5, "saturations": 1, "lightnesses": -1}, "opacity": {"x": 0, "y": 0, "bandwidth": 0}}
        # process the final answer line to get the action
        try:
            direction_json = json.loads(action_line)
        except json.JSONDecodeError:
            print("caught error when processing action, the action line is:", action_line)
            print("gpt_output: ", gpt_output)
            return -2, None
        
        direction = Direction()
        action = direction.set_from_json(direction_json)
        return action, direction
    
    @staticmethod 
    def process_cmp_output(gpt_output, img_id1=1, img_id2=2):
        import re  # 新增正则模块 
        
        if "final answer:" not in gpt_output.lower():
            return -2 
        
        final_answer_line = gpt_output.lower().split("final answer:")[1].strip()
        scores = final_answer_line.split() 
        
        # 强化过滤：正则匹配纯数字 
        scores = [s for s in scores if re.match(r'^\d+$',  s)]
        if not scores:
            return -2 
        
        # 精准捕获转换异常 
        try:
            scores = list(map(int, scores))
        except (ValueError, TypeError):
            print("caught error when processing scores, the final line is:", final_answer_line)
            print("gpt_output: ", gpt_output)
            return -2 
        
        # 计数逻辑放在try外部 
        count_1 = scores.count(1) 
        count_2 = scores.count(2) 
        
        if count_1 > count_2:
            return img_id1 
        elif count_2 > count_1:
            return img_id2 
        else:
            return -1 
        
    @staticmethod 
    def process_gaussian_output(gpt_output):
        import re  # 正则模块
        
        if "final answer:" not in gpt_output.lower():
            return -2 
        
        final_answer_line = gpt_output.lower().split("final answer:")[1].strip()
        
        # 使用正则表达式提取单个数字0或1
        match = re.search(r'\b([01])\b', final_answer_line)
        if not match:
            print("No valid answer (0 or 1) found in line:", final_answer_line)
            return -2
        
        # 返回匹配到的单个数字
        return int(match.group(1))
        
    def reset_text(self, text):
        """
        重置文本提示
        """
        # self.original_instruct = self.instruct
        self.instruct = text
        
    def reset_modification(self, text):
        """
        设置微调提示
        """
        self.modification = text
        
    def reset_format(self, type):
        """
        重置格式化提示
        """
        assert type == "quality" or type == "text" or type == "finetune" or type == "image"
        if type == "finetune":
            self.prompt_format = load_prompt(os.path.join(self.prompt_folder, 'prompt_edit_format.txt'))
        elif type == "image":
            self.prompt_format = load_prompt(os.path.join(self.prompt_folder, 'prompt_image_format.txt'))
        else:
            self.prompt_format = load_prompt(os.path.join(self.prompt_folder, 'prompt_format.txt'))
            
    def set_middle_image(self, middle_img):
        self.middle_img = middle_img.copy()
        
    def unset_middle_image(self):
        self.middle_img = None
        
    def format_prompt(self, prompt, volume_name="CT scan of human head", mode="quality"):
        assert mode == "quality" or mode == "text" or mode == "finetune" or mode == "gaussian" or mode == "image"
        
        if mode == "gaussian":
            #TODO 和真实的prompt对齐
            new_prompt = prompt
            new_prompt = new_prompt.replace("{DESCRIPTION}", self.instruct)
            return new_prompt
        
        new_prompt = prompt
        
        choices = list(map(int, self.modes[mode].split(",")))
        selected_aspects = [self.aspects[i - 1] for i in choices]
        
        formatted_aspects = [f"{i + 1}. {aspect['name']}: {aspect['description']}\n" for i, aspect in enumerate(selected_aspects)]
        formatted_aspects_string = "\n".join(formatted_aspects)
        formatted_answers = [f"{i + 1}. {aspect['name']}:The left one xxxx; The right one xxxx;\nThe left/right one is better or cannot decide\n" for i, aspect in enumerate(selected_aspects)]
        formatted_answers_string = "\n".join(formatted_answers)
        formatted_final_answers_front = [f"x" for i in range(len(selected_aspects))]
        formatted_final_answers_eg = [[f"{random.randint(1, 3)}" for i in range(len(selected_aspects))] for i in range(3)]
        formatted_final_answers_eg = [" ".join(formatted_final_answers_eg[i]) for i in range(3)]
        formatted_final_answers_string = " ".join(formatted_final_answers_front) + "(e.g., " + " / ".join(formatted_final_answers_eg) + ")"
        
        new_prompt = new_prompt.replace("{ASPECTS}", formatted_aspects_string)
        new_prompt = new_prompt.replace("{ANSWER_EXAMPLE}", formatted_answers_string)
        new_prompt = new_prompt.replace("{VOLUME_NAME}", volume_name)
        new_prompt = new_prompt.replace("{FINAL_ANSWER_EXAMPLE}", formatted_final_answers_string)
        
        if mode == "text":
            postfix_temp = 'Following is the text prompt from which these two volume rendered image are described:\n"{}"\nPlease compare these two image as instructed.'
            postfix = postfix_temp.format(self.instruct)
            new_prompt = new_prompt + "\n" + postfix
        
        if mode == "finetune":
            if self.instruct is None:
                # 删除 一句话
                new_prompt = new_prompt.replace("Following is the text prompt from which the original volume rendered image is described:", "")
                new_prompt = new_prompt.replace("{ORIGINAL_DESCRIPTION}", "")
            else:
                new_prompt = new_prompt.replace("{ORIGINAL_DESCRIPTION}", self.instruct)
            new_prompt = new_prompt.replace("{EDIT_INSTRUCTION}", self.modification)
            return new_prompt
        return new_prompt
    
    @staticmethod
    def image_to_base64_nparray(img):
        # 转换 PIL Image 为 numpy 数组
        img_array = np.array(img)

        # 将 numpy 数组转换为 JPEG 格式的字节流
        _, buffer = cv2.imencode('.jpg', img_array)

        # 直接转换为 Base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        return base64_str
    
    def llm_evaluate_2_image(self, img_base64, prompt, img_id1, img_id2, max_retries = 1):
        result = -2
        for attempt in range(max_retries):
            gpt_output = self.api.generate_text(prompt=prompt, imgbase64=img_base64)
            result = LLM_Evaluator.process_cmp_output(gpt_output, img_id1, img_id2)
            if result > 0: return result, gpt_output
        return result, gpt_output
    
    def llm_evaluate_gaussian(self, img_base64, prompt, max_retries = 1):
        for attempt in range(max_retries):
            gpt_output = self.api.generate_text(prompt=prompt, imgbase64=img_base64)
            action = LLM_Evaluator.process_gaussian_output(gpt_output)
            if action != -2: return action, gpt_output
        return None, gpt_output
    
    def llm_evaluate_1_mutate_image(self, img_base64, prompt, max_retries = 1):
        for attempt in range(max_retries):
            gpt_output = self.api.generate_text(prompt=prompt, imgbase64=img_base64)
            action, direction = LLM_Evaluator.process_mutate_output(gpt_output)
            if action != -2: return action, direction, gpt_output
        return "freeze", Direction(), gpt_output
    
    def evaluate_gaussian_image(self, img, base_img, volume_name, log_path = "", mute = True, return_output = True):
        if not img:
            print("Error: One or more input parameters (img) are None or empty in", log_path)
        prompt = self.format_prompt(prompt=self.prompt_format_gaussian, volume_name=volume_name, mode="gaussian")
        with open(os.path.join(log_path, f"prompt_gaussian.txt"), "w", encoding="utf-8") as prompt_file:
            prompt_file.write(prompt)
        img_base64, _ = combine_image(base_img, img, add_border=False)
        result, gpt_output = self.llm_evaluate_gaussian(img_base64=img_base64, prompt=prompt)
        if return_output == True:
            output = {"tf": 0, "gaussian_id": 0, "result": result, "gpt_output": gpt_output}
        else:
            output = None
        if result == 1:
            return "freeze", output
        else:
            return "no action", output
        
    def evaluate_visibility(self, img, log_path=None):
        """
        评估图像的可见度，根据指定指令让大模型返回可见度级别。
        
        Args:
            img: 要评估的图像对象(PIL Image)
            instruction: 提供给大模型的指令，如果为None则使用默认的指令
            log_path: 日志保存路径
            
        Returns:
            tuple: (可见度状态, 完整输出字典)
                可见度状态为 "not recognizable", "recognizable" 或 "clear"
        """
        if not img:
            print("Error: Input image is None or empty in", log_path)
            return "not recognizable", {"error": "Empty image"}
        
        # 确保日志路径存在
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # 将图像转换为base64
        img_base64 = self.image_to_base64_nparray(img)
        
        instruction = self.instruct
        
        # 构造提示 - 增加自然引导，避免直接进入instruction
        visibility_prompt = f"""
        You are an expert in analyzing volume rendered medical images. 
        Please evaluate the following image according to this instruction:
        
        {instruction}
        
        Based on your assessment, determine if the image fits the instruction.
        
        Your answer must follow this format:
        
        Analysis: [Your brief analysis of the visibility in the image]
        
        Final Assessment: [Only answer "not recognizable", "recognizable", or "clear"]
        - "not recognizable": if you cannot see the requested structures
        - "recognizable": if you can partially see them
        - "clear": if you can clearly see them
        """
        
        try:
            # 调用API获取评估结果
            gpt_output = self.api.generate_text(prompt=visibility_prompt, imgbase64=img_base64)
            
            # 解析输出，提取可见度评估结果
            visibility = self.process_visibility_output(gpt_output)
            
            # # 保存日志
            # if log_path:
            #     # 保存提示和响应
            #     with open(os.path.join(log_path, "visibility_prompt.txt"), "w", encoding="utf-8") as f:
            #         f.write(visibility_prompt)
                    
            #     with open(os.path.join(log_path, "visibility_response.txt"), "w", encoding="utf-8") as f:
            #         f.write(gpt_output)
                    
            #     # 保存图像
            #     img.save(os.path.join(log_path, "evaluated_image.png"))
                
            #     # 保存结构化结果
            #     with open(os.path.join(log_path, "visibility_result.json"), "w", encoding="utf-8") as f:
            #         json.dump({
            #             "visibility": visibility,
            #             "full_response": gpt_output
            #         }, f, indent=2)
            
            return visibility, {"response": gpt_output, "visibility": visibility}
            
        except Exception as e:
            print(f"评估可见度时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return "recognizable", {"error": str(e), "response": ""}
    def process_visibility_output(self, output):
        """
        从大模型输出中解析可见度评估结果
        
        Args:
            output: 大模型的完整输出文本
            
        Returns:
            str: "not recognizable", "recognizable" 或 "clear"
        """
        # 将输出转为小写以便匹配
        output_lower = output.lower()
        
        # 尝试找到"最终评估："后面的内容
        if "最终评估：" in output_lower or "最终评估:" in output_lower:
            # 提取最终评估部分
            final_part = output_lower.split("最终评估：")[-1].split("最终评估:")[-1].strip()
            
            # 匹配可能的结果
            if "not recognizable" in final_part:
                return "not recognizable"
            elif "clear" in final_part:
                return "clear"
            elif "recognizable" in final_part:
                return "recognizable"
        
        # 如果没有找到格式化的最终评估，尝试在整个文本中匹配
        if "not recognizable" in output_lower:
            return "not recognizable"
        elif "clear" in output_lower and "not clear" not in output_lower:
            return "clear"
        elif "recognizable" in output_lower:
            return "recognizable"
        
        # 默认返回"recognizable"作为最安全的选择
        print("无法从输出中解析可见度评估，使用默认值'not recognizable'")
        return "not recognizable"
    
    def compare_2_image(self, img1, img2, volume_name, mode, log_path = "", mute = True, return_output = True):
        if not all([img1, img2]):
            print("Error: One or more input parameters (img1, img2) are None or empty in", log_path)
        prompt = self.format_prompt(prompt=self.prompt_format, volume_name=volume_name, mode=mode)
        # save prompt to output file
        with open(os.path.join(log_path, f"prompt_{mode}.txt"), "w", encoding="utf-8") as prompt_file:
            prompt_file.write(prompt
        )
        # print("in mode", mode)
        if mode == "finetune" or mode == "image":
            if self.middle_img is not None:
                print("correct.")
            img1_base64, img2_base64 = combine_image(img1, img2, self.middle_img.copy())
        else:
            img1_base64, img2_base64 = combine_image(img1, img2)
        result1, gpt_output1 = self.llm_evaluate_2_image(img_base64=img1_base64, prompt=prompt, img_id1=1, img_id2=2)
        result2, gpt_output2 = self.llm_evaluate_2_image(img_base64=img2_base64, prompt=prompt, img_id1=2, img_id2=1)
        if return_output == True:
            output = {"img1": 0, "img2": 0, "result1": result1, "result2": result2, "gpt_output1": gpt_output1, "gpt_output2": gpt_output2}
        else:
            output = None
        #结果相同
        if result1 == result2:
            return state[result1], output
        #结果不同，并且都是非error和draw，说明两者选择相反，当作draw
        if result1 > 0 and result2 > 0:
            return "draw", output
        #选择不同，并且一方是error/draw，取另一方的选择
        if result1 > 0 and result2 < 0:
            return state[result1], output
        #选择不同，并且一方是error/draw，取另一方的选择
        if result1 < 0 and result2 > 0:
            return state[result2], output
        #选择不同，并且有一方是draw，当作draw
        if result1 == -1 or result2 == -1:
            return "draw", output
        #选择不同，并且都是error
        if log_path != "" and mute == False:
            LLM_Evaluator.log(log_path, img1_base64, img2_base64, gpt_output1, gpt_output2)
        return "error", output
    
    def mutate_1_image(self, img, volume_name, log_path = "", mute = True, return_output = True):
        prompt = self.format_prompt(prompt=self.prompt_format, volume_name=volume_name, mode="mutate")
        # save prompt to output file
        with open(os.path.join(log_path, f"prompt_mutate.txt"), "w", encoding="utf-8") as prompt_file:
            prompt_file.write(prompt
        )
        # 拼接两张图片
        img_base64 = image_to_base64_pil(img)
        action, directions, gpt_output = self.llm_evaluate_1_mutate_image(img_base64=img_base64, prompt=prompt)
        if return_output == True:
            output = {"img": 0, "action": action, "directions": directions, "gpt_output": gpt_output}
        else:
            output = None
            
        return directions, output
    
    @staticmethod
    def log(log_path, img1_base64, img2_base64, gpt_output1, gpt_output2):
        os.makedirs(log_path, exist_ok=True)
        # append gpt_output into log_path
        with open(os.path.join(log_path, "gpt_output"), "a", encoding="utf-8") as log_file:
            log_file.write(f"GPT output1: {gpt_output1}\n")
            log_file.write(f"GPT output2: {gpt_output2}\n")
        image_data1 = base64.b64decode(img1_base64)
        # 将图片保存到文件
        with open(os.path.join(log_path, "img1.png"), "wb") as file:
            file.write(image_data1)
        image_data2 = base64.b64decode(img2_base64)
        # 将图片保存到文件
        with open(os.path.join(log_path, "img2.png"), "wb") as file:
            file.write(image_data2)
        print(f"GPT output written to {log_path}\n")

if __name__ == "__main__":
    client = OpenAI(
        api_key="[API_KEY]",
        base_url="[URL]",
    )
    prompt_path = "/root/autodl-tmp/chatbot/prompt.txt"
    prompt = load_prompt(prompt_path)

    folder_path = "/root/autodl-tmp/chatbot/dataset/CQ500rdm2/generate/CQ500CT0 CQ500CT0/rgba_together"
    results = select_images_in_folder(client, prompt, folder_path, generate_num=3, max_num = 50)

    for result in results:
        # print(f"Image 1: {result['image1']}, Image 2: {result['image2']}, GPT Output: {result['gpt_output']}")
        save_txt = open('result_CQ500CT0 CQ500CT0.txt','a')
        save_txt.write(f"Image 1: {result['image1']}, Image 2: {result['image2']}, GPT Output1: {result['gpt_output1']}, GPT Output2: {result['gpt_output2']}\n")

    # results = process_image_test(client, prompt)

    # for result in results:
    #     # print(f"Image 1: {result['image1']}, Image 2: {result['image2']}, GPT Output: {result['gpt_output']}")
    #     save_txt = open('result_test.txt','a')
    #     save_txt.write(f"GPT Output: {result['gpt_output']}\n")