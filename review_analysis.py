from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
from konlpy.tag import Okt
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from krwordrank.sentence import summarize_with_sentences
import random
import re

# FastAPI 앱 생성
api = FastAPI()

# 글로벌 변수 및 초기화
stopwords_lst = ['등록된', '리뷰', '조금', '수', '것', '없습니다', '도움이', '정말', '살짝', '또', '너무', '잘', '같아요', '엄청', '바로', '한', '계속', '구매', '넣고', '먹기', '있어서', '다', '넘', '저는', '그냥', '맛이', '좋아요', 
                '아주', '펜슬', '좀', '좋고', '진짜', '완전', '있는', '때', '프로타주', '이', '좋아하는', '더', '좋은', '있습니다', '입니다', 
                '같아요', '있어', '좋은', '같습니다', '좋네요', '입니다', '있어요', '괜찮', '아니', '그런', 
                '같아', '좋습니다', '좋을', '있는데', '없어', '아니라', '좋은', '같은', '없는', '어요', '좋아', '좋앗', '입니', '있고', '없고', 
                '좋았', '좋습니', '생각',  '있을', '있습니', '있었', '아닌', '같습니', '습니', '니다', '정도']
okt = Okt()
ltokenizer = None  # 글로벌 토크나이저 변수

# 전처리 함수
def preprocessing(df, review_column, istokenize=True, pos_tags=None):
    processed_reviews = []
    for r in df[review_column]:
        sentence = re.sub(r'[^\w\s]', '', r)  # 특수문자 제거
        sentence = re.sub(r'\d+', '', sentence)  # 숫자 제거
        if len(sentence) == 0:
            continue

        if istokenize:
            tokens = ltokenizer.tokenize(sentence)
        else:
            tokens = [sentence]

        filtered_tokens = [token for token in tokens if len(token) > 1]
        processed_sentence = ' '.join(filtered_tokens)
        processed_reviews.append(processed_sentence)
    return pd.Series(processed_reviews)

# 키워드와 평균 평점 계산
def calculate_keyword_average_ratings(dataframe, keywords, review_column, rating_column):
    keyword_scores = []
    for keyword in keywords:
        keyword_reviews = dataframe[dataframe[review_column].str.contains(keyword, na=False)]
        avg_score = keyword_reviews[rating_column].mean()
        review_count = keyword_reviews.shape[0]
        keyword_scores.append({'Keyword': keyword, 'Average Rating': avg_score, 'Count': review_count})
    return pd.DataFrame(keyword_scores, columns=['Keyword', 'Average Rating', 'Count'])

# 문장 필터링 함수
def filter_sentences(df, keywords, min_length=30, max_length=60, num_samples=3):
    def split_sentences(text):
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)

    def get_sentences_with_keyword(review, keywords):
        sentences = split_sentences(review)
        return [sentence for sentence in sentences if any(keyword in sentence for keyword in keywords)]

    df["filtered_sentences"] = df["리뷰 내용"].apply(lambda x: get_sentences_with_keyword(str(x), keywords))
    all_sentences = [sentence for sentences in df["filtered_sentences"].dropna() for sentence in sentences]
    filtered_sentences = [sentence for sentence in all_sentences if min_length <= len(sentence) <= max_length]

    if len(filtered_sentences) >= num_samples:
        return random.sample(filtered_sentences, num_samples)
    else:
        return filtered_sentences

# API 엔드포인트
@api.post("/process/")
async def process_reviews(file: UploadFile, top_k: int = Form(10), min_length: int = Form(30), max_length: int = Form(60)):
    try:
        # 파일 로드
        df = pd.read_excel(file.file)
        df = df[df["리뷰 내용"] != "등록된 리뷰내용이 없습니다"]

        # WordExtractor 학습
        word_extractor = WordExtractor()
        word_extractor.train(df['리뷰 내용'].tolist())
        word_scores = word_extractor.extract()

        global ltokenizer
        scores = {word: score.cohesion_forward for word, score in word_scores.items()}
        ltokenizer = LTokenizer(scores=scores)

        # 리뷰 전처리
        df['preprocessed_review'] = preprocessing(df, review_column='리뷰 내용', istokenize=True)

        # 키워드 추출
        merged_reviews = '. '.join(df['preprocessed_review'].dropna())
        texts = merged_reviews.split('. ')
        keywords = summarize_with_sentences(texts, stopwords=stopwords_lst, num_keywords=top_k)[0]

        # 평균 평점 계산
        result_df = calculate_keyword_average_ratings(df, keywords, '리뷰 내용', '평점')

        # 문장 필터링
        full_merge_result = {}
        for i in range(len(result_df)):
            result = filter_sentences(df, result_df['Keyword'][i], min_length=min_length, max_length=max_length)
            full_merge_result[result_df['Keyword'][i]] = result

        return JSONResponse(content={
            "keywords": full_merge_result,
            "average_ratings": result_df.to_dict(orient="records")
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
