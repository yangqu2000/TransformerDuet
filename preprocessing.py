from music21 import *
import os

# chorales = corpus.search('bach', fileExtensions='xml')
#
# length = len(chorales)
#
# for i in range(length):
#
#     bwv = chorales[i].parse()
#
#     # bwv.write('musicxml', fp='1.xml')
#     # break
#
#     filename = bwv.metadata.title.split(".")[0] + '.midi'
#     bwv.write('midi', fp=os.path.join('dataset', filename))


files = os.listdir('test')

for midi_file in files:

    name = midi_file.split('.')[0]

    bwv = corpus.parse(name)
    bwv.write('musicxml', fp=f'{name}.xml')

