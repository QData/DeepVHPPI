
import re, collections

from pdb import set_trace as stop

def split(word): 
    return [char for char in word]  

def get_stats(vocab):
	pairs = collections.defaultdict(int)
	for word, freq in vocab.items():
		symbols = word.split()
		# symbols = split(word)
		for i in range(len(symbols)-1):
			pairs[symbols[i],symbols[i+1]] += freq
	return pairs


def merge_vocab(pair, v_in):
	v_out = {}
	# stop()
	bigram = re.escape(' '.join(pair))
	pr = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in v_in:
		w_out = pr.sub(''.join(pair), word)
		v_out[w_out] = v_in[word]
	return v_out


# vocab = {'l o w' : 5, 'l o w e r' : 2, 'n e w e s t':6, 'w i d e s t':3}
# vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,'n e w e s t </w>':6, 'w i d e s t </w>':3}
# vocab = {'low' : 5, 'lower' : 2, 'newest':6, 'widest':3}


vocab ={'n e w e r':1, 'w i d e r':1,'l o w':1, 'l o w e s t':1}


vocab = {'l o w l o w e r n e w e s t w i d e s t' : 1}


# vocab = {'WeevaluatetwomethodsofapplyingBPE:learningtwoindependentencodings,oneforthesource,oneforthetargetvocabulary,orlearningtheencodingontheunionofthetwovocabularies(whichwecalljointBPE.Theformerhastheadvantageofbeingmorecompactintermsoftextandvocabularysize,andhavingstrongerguaranteesthateachsubwordunithasbeenseeninthetrainingtextoftherespectivelanguage,whereasthelatterimprovesconsistencybetweenthesourceandthetargetsegmentation.IfweapplyBPEindependently,thesamenamemaybesegmenteddifferentlyinthetwolanguages,whichmakesitharderfortheneuralmodelstolearnamappingbetweenthesubwordunits.ToincreasetheconsistencybetweenEnglishandRussiansegmentationdespitethedifferingalphabets,wetransliteratetheRussianvocabularyintoLatincharacterswithISO-9tolearnthejointBPEencoding,thentransliteratetheBPEmergeoperationsbackintoCyrillictoapplythemtotheRussiantrainingtext.'}

num_merges = 3

for i in range(num_merges):
	
	pairs = get_stats(vocab)
	best = max(pairs, key=pairs.get)
	vocab = merge_vocab(best, vocab)

	print(best)

print('')
print(vocab)