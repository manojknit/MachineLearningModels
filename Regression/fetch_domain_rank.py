#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: yaminimuralidharen
"""

#!/usr/bin/env python
import urllib, sys, re
import urllib.request
import pdb
import io
import pandas as pd
import csv 
#with io.open("fake_news_subset.csv","r") as f:
df = pd.read_csv("fake.csv", nrows = 2000, encoding = "ISO-8859-1")
urldict = dict()
count =0
#pdb.set_trace()
#print(df.head(5))
#print(df)
#print(df[['site_url','domain_rank']])
for index,row in df.iterrows():
	print(row["site_url"], row["domain_rank"])
	if row["site_url"] not in urldict:
		urlval = row["site_url"]
		xml = urllib.request.urlopen('http://data.alexa.com/data?cli=10&dat=s&url=%s'%urlval).read()
	#	pdb.set_trace()
		try:
			rank = int(re.search(b'<POPULARITY[^>]*TEXT="(\d+)"', xml).groups()[0])
			urldict[urlval] = rank
			df.set_value(index,'domain_rank',urldict[urlval])
		#	pdb.set_trace()
			print("Your rank for %s is %d!\n" % (row["site_url"], rank))
		except: rank = -1
	else:
		count = count + 1
		rank = urldict.get(row["site_url"],"-1")
		df.set_value(index,'domain_rank',rank)
		print("Else part: Your rank for %s is %d!\n" % (row["site_url"], rank))
print(count)	
print(urldict)
print(df[['site_url','domain_rank']])
print(df.shape)
df.to_csv('fk_domain_rank.csv', index=False, encoding = "ISO-8859-1")
"""
	rows = csv.DictReader(f)
	for row in rows(5):
		#print(row["site_url"])
	"""