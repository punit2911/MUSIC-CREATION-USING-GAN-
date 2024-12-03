import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from music21 import converter, note, chord
from pathlib import Path
import matplotlib.pyplot as plt

SEQUENCE_LENGTH = 100
LATENT_DIMENSION = 100
BATCH_SIZE = 16
EPOCHS = 100
SAMPLE_INTERVAL = 10

def get_notes():
    """Get all the notes and chords from the MIDI files."""
    notes = []
    for file in Path("archive").glob("*.mid"):
        midi = converter.parse(file)
        print(f"Parsing {file}")
        notes_to_parse = midi.flatten().notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes

def prepare_sequences(notes, n_vocab):
    """Prepare the sequences used by the Neural Network."""
    pitchnames = sorted(set(notes))
    note_to_int = {note: num for num, note in enumerate(pitchnames)}
    
    network_input = []
    network_output = []
    for i in range(len(notes) - SEQUENCE_LENGTH):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[note] for note in sequence_in])
        network_output.append(note_to_int[sequence_out])

    network_input = np.reshape(network_input, (len(network_input), SEQUENCE_LENGTH, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output

class GAN:
    def __init__(self, seq_length, latent_dim, n_vocab):
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.n_vocab = n_vocab
        self.seq_shape = (self.seq_length, 1)

        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(noise)

        self.discriminator.trainable = False
        validity = self.discriminator(generated_seq)

        self.combined = Model(noise, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_discriminator(self):
        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        seq = Input(shape=self.seq_shape)
        validity = model(seq)
        return Model(seq, validity)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        seq = model(noise)
        return Model(noise, seq)

    def train(self, epochs, batch_size, sample_interval):
        notes = get_notes()
        n_vocab = len(set(notes))
        X_train, _ = prepare_sequences(notes, n_vocab)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        g_losses = []
        d_losses = []

        for epoch in range(epochs):
            # Training the Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, real)

            g_losses.append(g_loss)
            d_losses.append(d_loss[0])

            # Debugging and saving models
            if epoch % sample_interval == 0:
                print(f"{epoch}/{epochs} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}%] [G loss: {g_loss}]")
                self.save_loss_plot(g_losses, d_losses, epoch)

                # Save models
                if not os.path.exists("saved_models"):
                    os.makedirs("saved_models")
                self.generator.save(f"saved_models/generator_epoch_{epoch}.h5")
                self.discriminator.save(f"saved_models/discriminator_epoch_{epoch}.h5")

    def save_loss_plot(self, g_losses, d_losses, epoch):
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="Generator Loss")
        plt.plot(d_losses, label="Discriminator Loss")
        plt.title(f"Losses at Epoch {epoch}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_plot_epoch_{epoch}.png")
        plt.close()

if __name__ == '__main__':
    gan = GAN(seq_length=SEQUENCE_LENGTH, latent_dim=LATENT_DIMENSION, n_vocab=100)
    gan.train(epochs=EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)