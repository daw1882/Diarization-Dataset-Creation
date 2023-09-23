"""
mixture.py

Implementation of data structures and functions for creating audio mixtures and exporting them to rttm.

author: Wiley Matthews | wsm8855@rit.edu, Dade Wood | daw1882@rit.edu
"""

# native
import os
from typing import NamedTuple, Optional

# 3rd party
from pydub import AudioSegment

# local
import rttm


class Mixture:
    """
    Class for creating mixtures of noise and speakers as described in Kinoshita et al. Section 4.1.1
    Follows the builder pattern to build Mixtures.
    """
    class _SpeakerEntry(NamedTuple):
        position_abs_ms: int
        duration_ms: int
        src_fn: str
        speaker_name: str
        speaker_fn: str

        @property
        def end_position_abs_ms(self):
            return self.position_abs_ms + self.duration_ms

    def __init__(self, duration_ms: int):
        self.duration_ms = duration_ms
        self.audio_segment = AudioSegment.silent(duration=duration_ms)
        self.speaker_entries = []
        self.compiled = False

        self.rttm_df = None
        self.rttms = []
        self.fn = None
        self.rttm_path = None
        self.audio_path = None

    def add_speaker(
        self,
        audio_fn: str,
        speaker_name: str,
        speaker_fn: str,
        position_ms: int = 0,
        duration_ms: Optional[int] = None,
        audio_format: str = 'flac'
    ):
        """
        Add a new speaker file to the mixture.
        """
        if self.compiled:
            raise ValueError('Trying to call add_speaker after mixture has already been compiled')

        if position_ms is None:
            position_ms = 0

        # validate correctness
        assert isinstance(audio_fn, str)  # right type
        assert os.path.exists(audio_fn)  # file exists
        assert isinstance(speaker_name, str)  # right type
        assert ' ' not in speaker_name  # no whitespace

        # load audio segment
        audio_new = AudioSegment.from_file(audio_fn, format=audio_format)

        # determine actual length and location to overlay
        if duration_ms is not None and duration_ms < len(audio_new):
            duration_used_ms = duration_ms
            audio_used = audio_new[:duration_used_ms]
        else:
            duration_used_ms = len(audio_new)
            audio_used = audio_new

        # overlay and record data
        self.audio_segment = self.audio_segment.overlay(audio_used, position=position_ms)
        self.speaker_entries.append(self._SpeakerEntry(position_ms, duration_used_ms,  audio_fn, speaker_name, speaker_fn))

        return self

    def add_noise(
        self,
        noise_fn: str,
        position_ms: int = 0,
        duration_ms: Optional[int] = None,
        audio_format: str = 'flac'
    ):
        """
        Add a noise file to the mixture.
        """

        # input validation
        assert isinstance(noise_fn, str)  # right type
        assert os.path.exists(noise_fn)  # file exists
        assert position_ms >= 0  # valid value

        # load noise
        noise_segment = AudioSegment.from_file(noise_fn, format=audio_format)

        # trim as needed
        if duration_ms is not None and duration_ms < len(noise_segment):
            noise_segment = noise_segment[:duration_ms]

        # overlay into mixture
        self.audio_segment = self.audio_segment.overlay(noise_segment, position=position_ms)
        return self

    def compile(self, fn_no_ext: str):
        """
        Generate appropriate rttm data structure.
        """
        if self.compiled:
            raise ValueError('Trying to compile mixture twice')
        assert isinstance(fn_no_ext, str)  # right type

        # save filename for export
        self.fn = fn_no_ext

        # parse entries into rttm format
        sorted_entries = sorted(self.speaker_entries)  # sort by first occurrence
        for entry in sorted_entries:
            position_s = entry.position_abs_ms / 1000
            duration_s = entry.duration_ms / 1000
            speaker_name = entry.speaker_name
            self.rttms.append(rttm.RTTM(file_id=entry.speaker_fn,
                                   turn_onset=position_s,
                                   turn_duration=duration_s,
                                   speaker_name=speaker_name))

        # save rttms to df for easy handling
        self.rttm_df = rttm.rttms2df(self.rttms)

        # flag that mixture has been compiled - should not change from here
        self.compiled = True
        return self

    def export(self, rttm_dir: str, audio_dir: str, audio_format: str='wav'):
        """
        Export audio and rttm to specified locations.
        """
        if not self.compiled:
            raise ValueError('Trying to call export before mixture has been compiled')

        assert isinstance(rttm_dir, str)
        assert isinstance(audio_dir, str)

        if not os.path.exists(rttm_dir):
            print(f"rttm directory does not exist. Creating directory '{rttm_dir}'")
            os.makedirs(rttm_dir)
        if not os.path.exists(audio_dir):
            print(f"audio directory does not exist. Creating directory '{audio_dir}'")
            os.makedirs(audio_dir)

        # write rttm file
        self.rttm_path = os.path.join(rttm_dir, f'{self.fn}.rttm')
        with open(self.rttm_path, 'w') as rttm_file:
            rttm_file_str = rttm.rttms2file_str(self.rttms)
            rttm_file.write(rttm_file_str)

        # write audio file
        self.audio_path = os.path.join(audio_dir, f'{self.fn}.{audio_format}')
        self.audio_segment.export(self.audio_path, format=audio_format)

        return self


def run_test():
    # FILL OUT BELOW
    audio1_fn = ''
    audio2_fn = ''
    noise_fn = ''

    mixture = Mixture(duration_ms=10_000)
    mixture.add_noise(noise_fn, audio_format='wav')
    mixture.add_speaker(audio1_fn, 'spk1', position_ms=1500, duration_ms=3000, speaker_fn='A')
    mixture.add_speaker(audio2_fn, 'spk2', position_ms=3000, duration_ms=5000, speaker_fn='B')
    mixture.compile(fn_no_ext='test_mixture')
    mixture.export(rttm_dir='', audio_dir='', audio_format='wav')

    print('Done!')


if __name__ == '__main__':
    run_test()
