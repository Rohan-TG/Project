import requests
import periodictable



# url = 'https://www-nds.iaea.org/relnsd/v1/data?fields=levels&nuclides=135xe'
url = 'https://www-nds.iaea.org/relnsd/v1/data?fields=ground_states&nuclides=all'






response = requests.get(url)

print("Completed")

if response.status_code == 200:
	with open('livechart.csv', 'wb') as csvfile:
		csvfile.write(response.content)
		print("File downloaded")
else:
	print("Failed")