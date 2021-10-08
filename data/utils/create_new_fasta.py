 
import os

with open('uniprot_sprot.fasta') as f:
    content = f.readlines()

content = [x.strip() for x in content]


out_file = open('processed_uniprot_sprot.fasta','w')
flag = False
for line in content:
	if '>' in line:
		out_file.write('\n'+line+'\n')
	else:
		out_file.write(line)
out_file.close()

os.system('sed "1d" processed_uniprot_sprot.fasta > tmpfile; mv tmpfile processed_uniprot_sprot.fasta')
os.system('sed "/>sp/d" processed_uniprot_sprot.fasta > all_sequences.txt')


out_file = open('uniprot_sprot_human.fasta','w')
flag = False
for line in content:
	if 'OX=9606' in line:
		out_file.write('\n'+line+'\n')
		flag = True
	elif '>' in line:
		flag = False
	elif flag:
		out_file.write(line)
out_file.close()


os.system('sed "1d" uniprot_sprot_human.fasta > tmpfile; mv tmpfile uniprot_sprot_human.fasta')
os.system('sed "/>sp/d" uniprot_sprot_human.fasta > human_sequences.txt')