import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from pinecone import Pinecone
from dotenv import load_dotenv

# .env 파일에 저장된 API 키 환경 변수 로드
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "leadfit-shorts-index")

app = FastAPI(
    title="내몸교정 AI 코치 API",
    description="Gemini 2.5 + Pinecone 기반 체형 교정 코치 백엔드 API",
    version="1.0.0"
)

# 추천 영상 단일 객체 모델 (제목 + 유튜브 링크)
class RecommendedVideo(BaseModel):
    title: str
    url: str

# 앱에서 전달받을 요청 데이터 구조 (JSON Request)
class ChatRequest(BaseModel):
    user_question: str
    last_topic: str = ""
    recommended_titles: list[str] = []

# 앱으로 전달할 응답 데이터 구조 (JSON Response)
class ChatResponse(BaseModel):
    answer: str
    new_topic: str
    recommended_videos: list[RecommendedVideo] # 제목과 URL을 함께 전달

@app.get("/")
def read_root():
    return {"message": "LeadFit AI Coach API Server is Running!"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(req: ChatRequest):
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)

        user_question = req.user_question.strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="질문 내용이 비어있습니다.")

        # 거절/변경 키워드 체크
        rejection_keywords = ["다른", "말고", "아니", "별로", "다시", "다음", "새로운", "이거 말고"]
        is_rejection = any(kw in user_question for kw in rejection_keywords)

        if is_rejection and req.last_topic:
            search_query = req.last_topic
        else:
            search_query = user_question

        current_last_topic = search_query

        # 1) 검색어 임베딩 생성 (Pinecone 768 차원)
        embed_response = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=search_query,
            config={"output_dimensionality": 768},
        )
        query_vector = embed_response.embeddings[0].values

        # 2) Pinecone DB 검색
        search_results = index.query(
            vector=query_vector, top_k=15, include_metadata=True
        )

        # 3) 이미 추천한 영상 제외 필터링
        filtered_matches = []
        for match in search_results["matches"]:
            title = match["metadata"].get("title", "제목 없음")
            if title not in req.recommended_titles:
                filtered_matches.append(match)

        if not filtered_matches:
            filtered_matches = search_results["matches"][:2]
        else:
            filtered_matches = filtered_matches[:2]

        context_text = ""
        recommended_videos = []

        # 4) 제목 및 유튜브 URL 추출
        for match in filtered_matches:
            metadata = match.get("metadata", {})
            title = metadata.get("title", "제목 없음")
            raw_text = metadata.get("raw_text", "")
            
            # 메타데이터에서 url 또는 video_id, url_id를 찾아 전체 주소 생성
            url = metadata.get("url", "")
            if not url and "video_id" in metadata:
                url = f"https://www.youtube.com/watch?v={metadata['video_id']}"
            elif not url and "url_id" in metadata:
                url = f"https://www.youtube.com/watch?v={metadata['url_id']}"
            
            # 주소를 찾지 못한 경우 기본 주소 연결
            if not url:
                url = "https://www.youtube.com"

            context_text += f"\n[영상 제목: {title}]\n자막 내용: {raw_text}\n"
            recommended_videos.append(RecommendedVideo(title=title, url=url))

        # 5) Gemini 2.5 프롬프트 작성
        prompt = f"""
너는 전문 체형 교정 및 운동 코치 AI야.
현재 대화 주제는 '{current_last_topic}' 관련 내용이야.

[검색된 쇼츠 영상 정보]
{context_text}

[답변 지침]
1. 친절하고 전문적인 1:1 코치 어조로 말해 줘.
2. 사용자가 다른 영상/다른 방법을 원했다면, 주제('{current_last_topic}')를 벗어나지 않으면서 새로 검색된 영상의 핵심 원리와 동작을 자신감 있게 가이드해 줘.
3. "자막이 없다", "시스템 정보 부족" 같은 변명은 절대 하지 마.
4. 새로 추천하는 영상의 제목을 명확하게 언급해 줘.
5. 답변 끝에는 사용자가 동작을 이해했는지, 혹은 다음 질문이 있는지 편하게 물어봐 줘.

현재 사용자 입력: {user_question}
"""

        # 6) Gemini 2.5 Flash 모델 호출
        response = None
        for attempt in range(5):
            try:
                response = genai_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                break
            except Exception as e:
                if ("429" in str(e) or "503" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 4:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise e

        return ChatResponse(
            answer=response.text,
            new_topic=current_last_topic,
            recommended_videos=recommended_videos
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
