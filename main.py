from data import BachChoralesDataset
from model import *

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_loop(model, opt, loss_fn, dataloader):

    model.train()
    total_loss = 0

    for h_notes, m_notes in dataloader:

        h_notes = h_notes.to(device)
        m_notes = m_notes.to(device)

        m_notes_input = m_notes[:, :-1]
        m_notes_expect = m_notes[:, 1:]

        sequence_length = m_notes_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        pred = model(h_notes, m_notes_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)

        loss = loss_fn(pred, m_notes_expect)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for h_notes, m_notes in dataloader:
            h_notes = h_notes.to(device)
            m_notes = m_notes.to(device)

            m_notes_input = m_notes[:, :-1]
            m_notes_expect = m_notes[:, 1:]

            sequence_length = m_notes_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            pred = model(h_notes, m_notes_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, m_notes_expect)

            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):

    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in range(epochs):
        print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]

        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

        torch.save(model, f"model/Epoch_{epoch + 1}.pt")

    return train_loss_list, validation_loss_list


def main():
    data = BachChoralesDataset('dataset')

    train_data, val_data= torch.utils.data.random_split(data, [0.8, 0.2])

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=1,
        shuffle=True
    )

    model = Transformer(
        num_tokens=88 + 1 + 1 + 1, dim_model=512, num_heads=8, num_encoder_layers=3, num_decoder_layers=3, dropout_p=0.1
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-9)
    loss_fn = nn.CrossEntropyLoss()

    train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, val_dataloader, 20)

    x = list(range(1, 21))

    plt.style.use('seaborn-deep')
    plt.plot(x, train_loss_list, label='Train Loss')
    plt.plot(x, validation_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    main()
