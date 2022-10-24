import os


for p in os.listdir('.'):
  if p.endswith(' copy.json'):
    print(p)
    
    os.remove(p.replace(' copy', ''))
    os.rename(p, p.replace(' copy', ''))