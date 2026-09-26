import urllib.request; import urllib.error; 
try:
  res = urllib.request.urlopen(urllib.request.Request('http://localhost:8000/voice/test-call?to_number=%2B918318647325', method='POST'))
  print(res.read().decode())
except urllib.error.HTTPError as e:
  print(e.read().decode())
