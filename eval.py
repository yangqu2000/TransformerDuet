import pretty_midi
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn
import torch.nn.functional as F

import pretty_midi as pm
import os

from data import BachChoralesDataset

device = "cuda" if torch.cuda.is_available() else "cpu"


filenames = ['bwv103', 'bwv257', 'bwv339', 'bwv376', 'bwv7']


def convertToMidi(s_notes, b_notes, m_notes, filename):

    print(s_notes)

    midi = pm.PrettyMIDI()

    piano = pm.instrument_name_to_program('Acoustic Grand Piano')

    print("-" * 25, f"Convert soprano", "-" * 25)

    soprano = pretty_midi.Instrument(name='Soprano', program=piano)

    s_notes = s_notes[1:]
    t = 0
    i = 0
    while i < len(s_notes):

        start = t
        while i < len(s_notes) - 1 and s_notes[i] == s_notes[i + 1]:

            t += 0.125
            i += 1

        t += 0.125
        note = pm.Note(velocity=90, pitch=int(s_notes[i]), start=start, end=t)

        i += 1

        if note.pitch < 3:
            continue

        soprano.notes.append(note)

    print(soprano.notes)
    midi.instruments.append(soprano)

    print("-" * 25, f"Convert bass predict", "-" * 25)

    bass = pretty_midi.Instrument(name='Predict Bass', program=piano)

    t = 0
    i = 0
    while i < len(b_notes):

        start = t
        while i < len(b_notes) - 1 and b_notes[i] == b_notes[i + 1]:
            t += 0.125
            i += 1

        t += 0.125
        note = pm.Note(velocity=90, pitch=int(b_notes[i]), start=start, end=t)

        i += 1

        if note.pitch < 3:
            continue

        bass.notes.append(note)

    midi.instruments.append(bass)

    print("-" * 25, f"Convert ground truth", "-" * 25)

    bass_m = pretty_midi.Instrument(name='Ground-truth Bass', program=piano)

    m_notes = m_notes[1:]
    t = 0
    i = 0
    while i < len(m_notes):

        start = t
        while i < len(m_notes) - 1 and m_notes[i] == m_notes[i + 1]:
            t += 0.125
            i += 1

        t += 0.125
        note = pm.Note(velocity=90, pitch=int(m_notes[i]), start=start, end=t)

        i += 1

        if note.pitch < 3:
            continue

        bass_m.notes.append(note)

    midi.instruments.append(bass_m)

    print("-" * 25, f"Save midi file {filename}", "-" * 25)
    midi.write(filename + '.mid')


def test(model, dataloader):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    model.eval()

    with torch.no_grad():

        for i, (h_notes, m_notes) in enumerate(dataloader):
            h_notes = h_notes.to(device)
            m_notes = m_notes.to(device)

            m_notes = m_notes[:, :-1]

            y_input = torch.tensor([[1]], dtype=torch.long, device=device)

            while y_input.shape[1] < m_notes.shape[1]:
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(device)

                pred = model(h_notes, y_input, tgt_mask)
                pred = F.softmax(pred, dim=-1)
                print(pred.shape)
                print(pred[-1, :, :].topk(10)[1])

                next_note = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
                next_note = torch.tensor([[next_note]], device=device)

                y_input = torch.cat((y_input, next_note), dim=1)

            print(y_input)

            # sequence_length = m_notes.size(1)
            # tgt_mask = model.get_tgt_mask(sequence_length).to(device)
            # pred = model(h_notes, m_notes, tgt_mask)
            # pred_song = pred.topk(1)[1].view(-1)

            # convertToMidi(h_notes[0], pred_song, m_notes[0], filenames[i])

            break


model = torch.load('model/Epoch_2.pt')

data = BachChoralesDataset('test')

dataloader = DataLoader(
    dataset=data,
    batch_size=1,
    shuffle=False
)

test(model, dataloader)

