import multiprocessing as mp
import pickle

import librosa
import numpy as np
import pyaudio

from acc_dtw import accelerated_dtw


class Audio:  # TODO: Docstring
    def __init__(self, seconds, rate=44100, chunk=1024,
                 min_dist=100000, mfcc_path=None, labels_path=None):
        self.rate = rate
        self.chunk = chunk

        self.audio_len = int(rate / chunk * seconds)
        self.audio = [[]] * self.audio_len

        self.p = pyaudio.PyAudio()
        self.stream = self.open_stream()

        if mfcc_path and labels_path:  # TODO: Rename this shit
            self.min_dist = min_dist

            with open(mfcc_path, 'rb') as file:
                self.person_mfcc = pickle.load(file)
            with open(labels_path, 'rb') as file:
                self.person_labels = pickle.load(file)
            print(self.person_labels)

            self.min_shape_mfcc = min((mfcc.shape for mfcc in self.person_mfcc))

    def open_stream(self):
        stream = self.p.open(format=pyaudio.paFloat32,
                             channels=1,
                             rate=self.rate,
                             input=True,
                             frames_per_buffer=self.chunk)

        return stream

    def update_stream(self):  # TODO: asyncio maybe
        while True:
            try:
                byte_data = self.stream.read(self.chunk)
                data = np.frombuffer(byte_data, dtype=np.float32)
                # print(any(x != 0 for x in data))

                data = self.denoise_array(data)
                mfcc = self.array_to_mfcc(data, self.rate)

                # TODO: self.recognize_person() every self.audio_len

                # noinspection PyTypeChecker
                self.audio.append(mfcc)
                del self.audio[0]
                return self.audio

            except IOError as e:
                print(e)
                self.stream = self.open_stream()

    @staticmethod
    def denoise_array(array):  # TODO: Denoise
        return array

    @staticmethod
    def array_to_mfcc(array, rate=44100, max_pad_len=400):
        mfcc = librosa.feature.mfcc(array, rate)
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
        return mfcc

    def recognize_person(self, mfcc):
        pool = mp.Pool(processes=mp.cpu_count())

        pool_test_dataset = ((mfcc[:self.min_shape_mfcc[0], :self.min_shape_mfcc[1]],
                              mfcc2[:self.min_shape_mfcc[0], :self.min_shape_mfcc[1]])
                             for mfcc2 in self.person_mfcc)
        distances = pool.map(accelerated_dtw, pool_test_dataset)
        num_min = np.argmin(distances)

        print(self.person_labels[num_min], " - ", distances[num_min])


if __name__ == "__main__":
    audio = Audio(3, mfcc_path="mfcc_spiiras.pickle", labels_path="labels_spiiras.pickle")
    i = 0
    while True:
        s = audio.update_stream()
        i += 1

        if i == audio.audio_len:
            i = 0
            audio.recognize_person(s)
            print(i)
