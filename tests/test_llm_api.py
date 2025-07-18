import requests

prompt = input("You: ")

headers = {
	"Content-Type": "application/json"
	}

response = requests.post(
	"http://localhost:11434/api/generate",
	headers=headers,
	json={
		"model": "phi",
		"prompt": prompt,
		"stream": False
	}
)

print("\nK-14T:", response.json()["response"])
