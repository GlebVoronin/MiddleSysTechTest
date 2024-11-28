import requests

api_url = 'http://192.168.1.97:5002'
api_key = 'test28112024'
query = 'оперативная память'


# нормализация текста
print(requests.post(f'{api_url}/process', json={'text': query, 'api_key': api_key}).json()['result'])

print('\n\n')

# имитация ошибок при вызове api
print(requests.post(f'{api_url}/process', json={'text': query, 'api_key': '1'}).json()['Error'])
print(requests.post(f'{api_url}/process', json={'text': query}).json()['Error'])
print(requests.post(f'{api_url}/process', json={'text': '', 'api_key': api_key}).json()['Error'])
print(requests.post(f'{api_url}/process1', json={'text': query}).json()['Error'])
print(requests.post(f'{api_url}/search', json={'text': query, 'api_key': '1'}).json()['Error'])
print(requests.post(f'{api_url}/search', json={'text': query}).json()['Error'])
print(requests.post(f'{api_url}/search', json={'text': '', 'api_key': api_key}).json()['Error'])
print(requests.post(f'{api_url}/search1', json={'text': query}).json()['Error'])
