from services.groq_client import GroqClient

client = GroqClient()

response = client.generate("Explain cybersecurity risk in 2 lines")

print(response)