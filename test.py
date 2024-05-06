import requests, json

#URL 설정
url = "http://127.0.0.1:8080"

#보낼 데이터를 json.dumps()를 이용하여 json으로 변환
body = json.dumps({
    "id" : 3,
    "name" : "감자",
    "weight" : 320
})

#header 설정으로, json 통신 셋팅
header = {
    "Content-Type" : "application/json"
}

#.content를 이용한 요청 후 응답을 res에 저장
res = requests.post(url, body, headers=header).content

#응답 출력
print(res)