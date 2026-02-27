from requests import Session

s = Session()

jsn = {'query': 'Quero enviar uma requisição para um chatbot'}
response = s.post('http://127.0.0.1:5000/query', json=jsn)
print(response.json())