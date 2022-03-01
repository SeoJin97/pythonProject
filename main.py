import re
import numpy as np
import konlpy
import nltk

# 1)define hangule function
# target = "컵 핀 개 관 카드 캐시백 나이키 나이키 565656 @@# abf " # sample
target = "요리하다 고운 천일염 565656 @@# abf " # sample
stop_words = []
def hanguel_func(target) :
    # 1) 한글만 남기기
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기를 제외한 모든 글자
    result = hangul.sub(' ', target) # 한글과 띄어쓰기를 제외한 모든 부분을 제거

    # 2) 불필요한 단어 제거 (불용어 제거)
    stop_words = "카드 캐시백 무료 배송 무료배송 당일발송 무료견적 청구할인 천원 세트 케이스포함 신한 이월상품 일반용 포함 일반 "
    stop_words += "현대 현대백화점 롯데백화점 신세계백화점 백화점 하프클럽 갤러리아 행사 수원점 플라자 신세계인천점 신세계센텀점 "
    stop_words += "나비 환영 여성 남성 성인용 추가 플러스 인용 개월 할인 "
    stop_words += "디자인 컬러 검정 블랙 레드 빨강 옐로우 노랑 실버 네이비 그린 블루 "
    stop_words=stop_words.split(' ')
    # word_tokens = word_tokenize(result)
    word_tokens = result.split(' ')
    result_2 = []
    for w in word_tokens:
        if w not in stop_words:
            result_2.append(w)

    # 3) 중복 워드 없애기
    result_2 = list(set(result_2))

    # 4) 한글자 미만 줄이기
#     save_words = "컵 핀 넥 퀸 젤 캡 볼 힐 백 꽃 빔 꿀 폰 옷 천 면 찜 흙 껌 솥 떡"
    save_words = " "
    save_words = save_words.split(' ')
    result_3 = []
    for i in range(len(result_2)) :
        if len(result_2[i]) == 1:
            if result_2[i] in list(save_words) :
                result_3.append(result_2[i])
            else:
                pass
        else:
            result_3.append(result_2[i])
    print(result_3)

    # 5) 형용사 없애기(형태소 분석)

    for i, document in enumerate(result_3):
        okt = konlpy.tag.Okt()
        clean_words = []
        for word in okt.pos(document, stem=True):  # 어간 추출
            if word[1] in ['Noun', 'Verb', 'Adjective']:  # 명사, 동사, 형용사
                clean_words.append(word[0])
        #print(clean_words)
        document = ' '.join(clean_words)
        #print(document)
        result_3[i] = document

    # 9) 최종 전처리 결과 return
    answer = ""
    for i in range (0, len(result_3)):
        answer += result_3[i] + " "

    print(answer)
    return answer
hanguel_func(target) # check

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()