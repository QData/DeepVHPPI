
count =0

genome_file = '/bigtemp/jjl5sw/hg38/all_chars_full_uppercase.txt'

unigram_dict = {'A':0,'C':0,'G':0,'T':0,'N':0}
bigram_dict = {}
trigram_dict = {}
fourgram_dict = {}
fivegram_dict = {}
sixgram_dict = {}
sevengram_dict = {}
eightgram_dict = {}

prev1 = False
prev2 = False
prev3 = False
prev4 = False
prev5 = False
prev6 = False
prev7 = False

with open(genome_file,'r') as f:
	while True: 
		c = f.read(1)
		
		if not c:
			print("End of file")
			break
		unigram_dict[c]+=1

		if prev1:
			bigram = prev1+c
			if bigram not in bigram_dict:
				bigram_dict[bigram] = 0
			bigram_dict[bigram] += 1

		if prev2:
			trigram = prev2+prev1+c
			if trigram not in trigram_dict:
				trigram_dict[trigram] = 0
			trigram_dict[trigram] += 1

		if prev3:
			fourgram = prev3+prev2+prev1+c
			if fourgram not in fourgram_dict:
				fourgram_dict[fourgram] = 0
			fourgram_dict[fourgram] += 1

		if prev4:
			fivegram = prev4+prev3+prev2+prev1+c
			if fivegram not in fivegram_dict:
				fivegram_dict[fivegram] = 0
			fivegram_dict[fivegram] += 1

		if prev5:
			sixgram = prev5+prev4+prev3+prev2+prev1+c
			if sixgram not in sixgram_dict:
				sixgram_dict[sixgram] = 0
			sixgram_dict[sixgram] += 1
		
		if prev6:
			sevengram = prev6+prev5+prev4+prev3+prev2+prev1+c
			if sevengram not in sevengram_dict:
				sevengram_dict[sevengram] = 0
			sevengram_dict[sevengram] += 1

		if prev7:
			eightgram = prev7+prev6+prev5+prev4+prev3+prev2+prev1+c
			if eightgram not in eightgram_dict:
				eightgram_dict[eightgram] = 0
			eightgram_dict[eightgram] += 1

		
		prev7 = prev6
		prev6 = prev5
		prev5 = prev4
		prev4 = prev3
		prev3 = prev2
		prev2 = prev1
		prev1 = c

		# count+=1
		# if count == 100:
		# 	break

sorted_unigram_dict = sorted(unigram_dict.items(), key=lambda kv: kv[1],reverse=True)
sorted_bigram_dict = sorted(bigram_dict.items(), key=lambda kv: kv[1],reverse=True)
sorted_trigram_dict = sorted(trigram_dict.items(), key=lambda kv: kv[1],reverse=True)
sorted_fourgram_dict = sorted(fourgram_dict.items(), key=lambda kv: kv[1],reverse=True)
sorted_fivegram_dict = sorted(fivegram_dict.items(), key=lambda kv: kv[1],reverse=True)

with open('unigram_count.csv','w') as f: 
	for pair in sorted_unigram_dict:
		word = pair[0]
		count = pair[1]
		f.write("%s,%d\n"%(word,count))

with open('bigram_count.csv','w') as f: 
	for pair in sorted_bigram_dict:
		word = pair[0] 
		count = pair[1]
		f.write("%s,%d\n"%(word,count))

with open('trigram_count.csv','w') as f: 
	for pair in sorted_trigram_dict:
		word = pair[0] 
		count = pair[1]
		f.write("%s,%d\n"%(word,count))

with open('fourgram_count.csv','w') as f: 
	for pair in sorted_fourgram_dict:
		word = pair[0] 
		count = pair[1]
		f.write("%s,%d\n"%(word,count))

with open('fivegram_count.csv','w') as f: 
	for pair in sorted_fivegram_dict:
		word = pair[0] 
		count = pair[1]
		f.write("%s,%d\n"%(word,count))

