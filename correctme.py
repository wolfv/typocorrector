
import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import cKDTree as KDTree

from pprint import pprint
import math
layout = """qwertyuiop
.asdfghjkl
..zxcvbnm
"""

from marisa_trie import RecordTrie

class Correct:

	def get_vec_mapping(self, layout):
		max_line_length = max([len(line) for line in layout.split()])
		max_height = len(layout.split())

		mapping = {}
		y_pos = max_height  # start at top
		x_increment = 2

		for line in layout.split():
			# x_pos = -max_line_length / 2 # start left

			x_pos = 0
			for letter in line.strip():
				mapping[letter] = np.array([x_pos, y_pos])
				if letter == '.':
					x_pos += 1
				else:
					x_pos += x_increment
			y_pos -= 1
			
		print(mapping)
		self.mapping = mapping

	def get_vector_for_word(self, word):
		# mlist = [1,  3, 5,  7, 11, 13, 17, 19, 23, 27, 29, 31, 37, 41, 43, 47, 49, 53]
		multiplier = 1
		vector = np.zeros(3)
		prev_letter = None
		prevprev_letter = None
		basefct = math.sin
		def get_base_vec(idx, letter):
			letterindx = ord(letter) - ord('a')
			i1 = (idx + letterindx - 1) / 10.0
			i2 = (idx + letterindx) / 10.0
			cos = math.cos
			sin = math.sin
			return np.array([cos(i2) - cos(i1), sin(i2) - sin(i1)])
		for idx, letter in enumerate(word):
			try:
				if idx == 0:
					# multiplier = idx
					sep = 100
				else:
					# multiplier = math.e ** (float(idx) / 8.0)
					# multiplier = math.e ** (float(idx) / 8.0)
					# multiplier = 1
					# sep = math.e ** (float(idx) / 8.0)
					sep = 1
				# z = np.linalg.norm(self.mapping[letter]) * (idx + 1)
				if prevprev_letter:
					# z = np.abs(np.arctan2(self.mapping[letter][1] - self.mapping[prev_letter][1], 
					# 			   self.mapping[letter][0] - self.mapping[prev_letter][0]))
					# z = np.dot( self.mapping[prev_letter], self.mapping[letter])

					v1 = self.mapping[prev_letter] - self.mapping[prevprev_letter]
					v2 = self.mapping[prev_letter] - self.mapping[letter]
					z = np.arccos(np.dot(v1, v2)/ (np.linalg.norm(v1) * np.linalg.norm(v2)))
					if np.isnan(z):
						z = 0
					# z += math.pi
				else:
					z = 0.0
				vector += np.hstack((self.mapping[letter] * sep + get_base_vec(idx, letter) * 3, z))

				# multiplier = 1
				prevprev_letter = prev_letter
				prev_letter = letter
			except KeyError as e:
				print("%s doesn't exist in mapping :(" % letter)
		return vector


	def get_vectors_from_dict(self):

		self.vectors = {}
		
		for word in self.dict:
			try:
				self.vectors[word] = self.get_vector_for_word(word)
			except KeyError as e:
				print("%s can't be used as key :(" % word)
			# multiplier = 1
			# for letter in word:
			# 	try:
			# 		self.vectors[word] += self.mapping[letter] * multiplier
			# 		multiplier += 0.33
			# 	except KeyError as e:
			# 		print("%s doesn't exist in mapping :(" % letter);


	def match(self, word):
		matchvec = self.get_vector_for_word(word) # - np.array([-1, 0])
		print("\n\nTesting %s\nVector: %r" % (word, matchvec))
		match_dists, match_idxs = self.kdtree.query(matchvec, k=50)
		results = []
		for idx in match_idxs:
			results.append(self.vectors.items()[idx])

		return match_idxs, results


	def score(self, word, context=None):
		# score word
		# context is previous words in order!
		wscore = self.score_trie.get(word)
		if not wscore:
			wscore = 0
		gram2score = 1
		gram3score = 1
		if context:
			context = context.split()
			if len(context) > 1:
				gram2score = self.score_trie.get(word + " " + context[-1])
			if len(context) > 2:
				gram3score = self.score_trie.get(word + " " + context[-1] + " " + context[-2])

		if wscore: wscore = wscore[0]
		if type(gram2score) == tuple: wscore = gram2score[0]
		if type(gram3score) == tuple: wscore = gram3score[0]

		return wscore * gram2score * gram3score

	def score_and_sort_matches(self, matches, context):
		scores = []
		for m in matches:
			scores.append(self.score(m[0], context))

		res = zip(matches, scores)
		return sorted(res, key=lambda x: x[1])

	def __init__(self):
		vals = []
		with open('./en_wordlist.combined') as infile:
			items = l.split(',')
			# print(items)
			w = items[0].split('=')[1]
			if items[0].startswith('  '):
				print(l)
				continue
			f = int(items[1].split('=')[1])
			vals.append(w)

		# self.dict = [word.strip() for word in infile.readlines()]
		self.dict = vals

		self.vectors = {}
		self.mapping = {}
		self.score_trie = RecordTrie(fmt='<H')
		with open('trie.marisa') as ftrie:
			self.score_trie.read(ftrie)

		print(self.get_vec_mapping(layout))
		self.get_vectors_from_dict()
		# print(self.vectors)

		allvecs = [arr for arr in self.vectors.itervalues()]
		allvecsarr = np.array(allvecs)

		self.kdtree = KDTree(data=allvecsarr)

		print("#####\n\n\n")
		print(self.vectors['test'])
		print("####")
		pprint(self.match('test'))
		pprint(self.score_and_sort_matches(self.match('teat')[1], 'nice'))

		pprint(self.match('tedt')[1])
		pprint(self.match('perspecitev')[1])
		pprint(self.match('angle')[1])
		pprint(self.match('angel')[1])

		print("#####\n\n\n")

		print(self.vectors['news'])
		print("####")
		pprint(self.match('newa')[1])
		pprint(self.match('newr')[1])
		pprint(self.match('newst')[1])
		ms = self.match('newst')[1]
		v = self.get_vector_for_word('newst')
		pprint(sorted(ms, key=lambda x: abs(x[1][2] - v[2])))

		pprint(self.match('obascure')[1])
		pprint(self.match('obscure')[1])
		pprint(self.match('relativetiy')[1])
		pprint(self.match('absolutealy')[1])
		pprint(self.match('porcealitn')[1])

		print(allvecsarr)
		plt.scatter(allvecsarr[:, 0], allvecsarr[:, 1])
		plt.show()


if __name__ == '__main__':
	Correct()