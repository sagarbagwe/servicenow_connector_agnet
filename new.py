import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta

INSTANCE = "https://dev206840.service-now.com"
USER = "sn_user"   # your username
PASS = "Google@2562"

# Calculate the date 7 days ago in ServiceNow's date format
seven_days_ago = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")

url = f"{INSTANCE}/api/now/table/incident"
params = {
    'sysparm_limit': 5,
    'sysparm_query': f"sys_created_on>={seven_days_ago}^ORDERBYDESCsys_created_on"
}

response = requests.get(url, auth=HTTPBasicAuth(USER, PASS), params=params)
data = response.json()

print("📅 Incidents from the last 7 days:")
for i, rec in enumerate(data['result'], 1):
    print(f"{i}. {rec['number']} - {rec['short_description']}")
