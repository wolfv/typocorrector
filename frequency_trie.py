import marisa_trie

# trie = marisa_trie.RecordTrie()
import codecs
import sys
keys = []
values = []

import IPython

def import_android(file):
	with open(file) as f:
		# discard first line
		a = f.readline()

		for l in f.readlines():
			items = l.split(',')
			# print(items)
			w = items[0].split('=')[1]
			if items[0].startswith('  '):
				print(l)
				continue
			f = int(items[1].split('=')[1])
			keys.append(w)
			values.append((f,))

def import_2_gram(file, n=2):
	with codecs.open(file, encoding='ISO-8859-2') as f:
		for l in f.readlines():
			# print(l)
			items = l.split('\t')
			f = int(items[0])
			if(f > 65535):
				print(l)
				f = 65535
			key = items[2].strip() + " " + items[1].strip()
			keys.append(key)
			values.append((f, ))

def import_3_gram(file, n=3):
	with codecs.open(file, encoding='ISO-8859-2') as f:
		idx = 0
		for l in f.readlines():
			# print(l)
			items = l.split('\t')
			f = int(items[0])
			if(f > 65535):
				print(l)
				f = 65535
			key = items[3].strip() + " " + items[2].strip() + " " + items[1].strip()
			# print(key)
			keys.append(key)
			idx += 1
			values.append((f, ))

class Trie():

	def __init__(self, lang='en'):
		with open('trie.marisa') as fi:
			trie.write(fi)

	def search(self, word):
		return self.trie.get(word)

if __name__ == '__main__':
	fmt = "<H"
	if len(sys.argv) > 1:
		with open(sys.argv[1]) as ftrie:
			trie = marisa_trie.RecordTrie(fmt).read(ftrie)
	else:
		import_android('./en_wordlist.combined')
		import_2_gram('./w2_.txt')
		import_3_gram('./w3_.txt')
		trie = marisa_trie.RecordTrie(fmt, zip(keys, values))

	# while True:	# infinite loop
	# 	n = input("\nSearch Trie: ")
	# 	if n == "exit":
	# 		break  # stops the loop
	# 	else:
	# 		print(trie.get(n))
	# 		# etc.
	with open('trie.marisa', 'w') as fo:
		trie.write(fo)

	IPython.embed()