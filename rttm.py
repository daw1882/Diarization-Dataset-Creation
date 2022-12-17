"""
rttm.py

Implementation of data structures and functions for RTTM files.
RTTM is the data structure used to represent diarization labels for a text file.
https://stackoverflow.com/questions/30975084/rttm-file-format

An example line in an RTTM file looks like this: SPEAKER file2 1 2 0.5 <NA> <NA> john <NA> <NA>

authors: Wiley Matthews | wsm8855@rit.edu
"""

# 3rd party
import pandas as pd

# local
from typing import NamedTuple, List, Tuple

NA_DEFAULT = '<NA>'
TYPE_DEFAULT = 'SPEAKER'
CHANNEL_ID_DEFAULT = 1
ORTHOGRAPHY_FIELD_DEFAULT = NA_DEFAULT
SPEAKER_TYPE_DEFAULT = NA_DEFAULT
CONFIDENCE_SCORE_DEFAULT = NA_DEFAULT
SIGNAL_LOOKAHEAD_TIME_DEFAULT = NA_DEFAULT

DF_COLS = ['file_id', 'turn_onset', 'turn_duration', 'speaker_name']


class RTTM(NamedTuple):
    """
    Data structure that represents the fields of an entry in an RTTM file. AKA the time of a single speaker segment.

    Some fields always take the same value, so they are not required to be defined but will show up when the string
    that should go into a file is generated (see *_DEFAULT constants above).

    All string fields must not have any whitespace.
    """
    file_id: str
    turn_onset: float
    turn_duration: float
    speaker_name: str

    def file_str(self) -> str:
        """
        Create the string representation of the rttm entry that should go in an rttm file (newline not included)
        """
        return ' '.join([str(f) for f in self.fields])

    @property
    def fields(self) -> Tuple:
        """
        Return all of the fields of the rttm entry as a tuple. (All fields in their native data type)
        """
        return (
            TYPE_DEFAULT,
            self.file_id,
            CHANNEL_ID_DEFAULT,
            self.turn_onset,
            self.turn_duration,
            ORTHOGRAPHY_FIELD_DEFAULT,
            SPEAKER_TYPE_DEFAULT,
            self.speaker_name,
            CONFIDENCE_SCORE_DEFAULT,
            SIGNAL_LOOKAHEAD_TIME_DEFAULT
        )


def rttms2file_str(rttms: List[RTTM]):
    """
    Convert a list of rttm entries into a single string that can be written to a file.
    """
    rttm_strs = '\n'.join([rttm.file_str() for rttm in rttms])
    return rttm_strs


def file_str2rttm(file_str: str):
    """
    Parse an rttm entry object from its string representation read from a file.
    """
    fields = file_str.strip().split(' ')
    file_id = fields[1]
    turn_onset = float(fields[3])
    turn_duration = float(fields[4])
    speaker_name = fields[7]

    rttm = RTTM(file_id=file_id, turn_onset=turn_onset, turn_duration=turn_duration, speaker_name=speaker_name)
    return rttm


def rttms2df(rttms):
    if len(rttms) > 0:
        df = pd.DataFrame(data=rttms, columns=DF_COLS)
    else:
        df = pd.DataFrame({col: [] for col in DF_COLS})  # empty df with correct cols
    return df


def run_test():
    """
    Run some basic tests to make sure that things are working correctly.
    """
    rttm1 = RTTM('file1', 1.5, 2, 'jane')
    rttm2 = RTTM('file1', 2, 0.5, 'john')

    print('Tuple form')
    print('rttm1:', rttm1)
    print('rttm2:', rttm2)
    print()

    print('File form')
    print('rttm1:', rttm1.file_str())
    print('rttm2:', rttm2.file_str())
    print()

    ex_input_str = 'SPEAKER file2 1 2 0.5 <NA> <NA> john <NA> <NA>\n'  # example line from file
    rttm3 = file_str2rttm(ex_input_str)

    print('Example input:', ex_input_str, end='')
    print('rttm3:', rttm3)


if __name__ == '__main__':
    run_test()
