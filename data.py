import pretty_midi as pm
import numpy as np
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torchaudio

from torch.nn.utils.rnn import pad_sequence

import librosa
import librosa.display

D_DIM = 256


def addNotes(notes):
    note_list = [1]

    t = 0
    for note in notes:

        if note.start != t:
            hold_dur = note.start - t

            if hold_dur % 0.125 != 0:
                return None, -1

            length = int(hold_dur / 0.125)
            for _ in range(length):
                note_list.append(0)

        dur = note.end - note.start

        if dur % 0.125 != 0:
            return None, -1

        length = int(dur / 0.125)
        for _ in range(length):
            note_list.append(note.pitch)

        t = note.end

    return note_list, None


class BachChoralesDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.h_notes = []
        self.m_notes = []
        self.loadData()

    def __len__(self):
        return len(self.h_notes)

    def __getitem__(self, idx):
        h_note = self.h_notes[idx]
        m_note = self.m_notes[idx]

        return torch.tensor(h_note), torch.tensor(m_note)

    def loadData(self):

        print("-----------Start loading file------------")

        files = os.listdir(self.data_dir)

        for i, midi_file in enumerate(files):

            print(midi_file)
            midi_data = pm.PrettyMIDI(os.path.join(self.data_dir, midi_file))

            for instrument in midi_data.instruments:

                if instrument.name == "Soprano":

                    h_note, err = addNotes(instrument.notes)
                    if err is not None:
                        continue

                    # End of Song <EOS> token = 2
                    h_note.append(2)

                    self.h_notes.append(h_note)

                elif instrument.name == "Bass":

                    m_note, err = addNotes(instrument.notes)
                    if err is not None:
                        self.h_notes.pop()
                        continue

                    # End of Song <EOS> token = 2
                    m_note.append(2)

                    self.m_notes.append(m_note)

        print("-----------Finish loading file------------")

        # return midis, audios, labels


# data = BachChoralesDataset('dataset')