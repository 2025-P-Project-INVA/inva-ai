"""Gemini API Client Wrapper"""
from google import genai
from google.genai import types
import os


class GeminiClient:
    """Gemini API 클라이언트 래퍼"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """
        Args:
            api_key: Gemini API 키 (없으면 환경변수 GEMINI_API_KEY 사용)
            model_name: 사용할 모델명
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)
    
    def generate(self, prompt: str) -> str:
        """프롬프트를 받아 Gemini 응답 텍스트 반환
        
        Args:
            prompt: 입력 프롬프트
            
        Returns:
            생성된 텍스트
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
