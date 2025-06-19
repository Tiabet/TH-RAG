import os
import json
from openai import OpenAI
from prompt.topic_prompt import TOPIC_PROMPT  # 여기에 your prompt 템플릿이 문자열로 정의되어 있어야 합니다

# === 설정 ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("환경 변수 OPENAI_API_KEY를 설정해야 합니다.")

MODEL_NAME = "gpt-4o-mini"


def extract_topics_subtopics(
    query: str,
    client: OpenAI,
) -> list[dict]:
    """
    Returns:
      [
        {"topic": "Topic1", "subtopics": ["Sub1", ..., "Sub10"]},
      ]
    """
    prompt = TOPIC_PROMPT.replace("{question}", query)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"모델 응답에서 JSON 파싱 실패: {e}\n---\n{content}")

    topics = data.get("topics")
    if not isinstance(topics, list) or len(topics) != 1:
        raise ValueError(f"예상과 다른 토픽 개수: {topics}")
    for t in topics:
        if "topic" not in t or "subtopics" not in t or len(t["subtopics"]) != 10:
            raise ValueError(f"서브토픽 형식 오류: {t}")

    return topics

# 실행 코드 (메인 가드 없이)
client = OpenAI(api_key=OPENAI_API_KEY)
query = "What are the steps involved in extracting and handling honey, and why is it important to strain the honey after extraction?"
topics = extract_topics_subtopics(query, client)
print(json.dumps(topics, ensure_ascii=False, indent=2))
