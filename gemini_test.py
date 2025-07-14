import os
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 키 불러오기
api_key = os.getenv("GEMINI_API_KEY")

# API 키 설정
genai.configure(api_key=api_key)

# 모델 초기화
model = genai.GenerativeModel("gemini-2.0-flash")

# 프롬프트 요청
prompt = "Explain the concept of Occam's Razor and provide a simple, everyday example."
response = model.generate_content(prompt)

# 결과 출력
print(response.text)
