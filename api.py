from flask import Blueprint, render_template, request, jsonify,Flask
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import sys
#from script import SequencesAnalyzer

app = Flask(__name__)

# Import ScoringSystems and SequencesAnalyzer here

SCORES_CSV = './scores.csv'
EDIT_COST_CSV = './edit_cost.csv'

class ScoringSystems:
    '''Responsible for returning proper scoring/edit cost values for any letter combination'''

    def __init__(self, match: int=1, mismatch: int=-1, gap: int=-1) -> None:
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.custom_scoring = None

    def load_csv(self, filename: str) -> None:
        self.custom_scoring = pd.read_csv(filename, header=0, index_col=0, sep=' ')

    def _default_scoring(self, a: str, b: str) -> int:
        '''Checks if there's a match/mismatch/gap for letters a and b'''
        if a == b:
            return self.match
        elif a == '-' or b == '-':
            return self.gap
        return self.mismatch

    def score(self, a: str, b: str) -> int:
        '''This method should be used by algorithms'''
        assert isinstance(a, str) and isinstance(b, str)
        assert len(a) == 1 and len(b) == 1

        # Use CSV file
        if self.custom_scoring is not None:
            try:
                return self.custom_scoring[a][b]
            except KeyError:
                print(f'WARNING: Key not found. You using defaults: {self.__str__}')
                # In case some letter was not present in CSV file, use default scoring value
                return self._default_scoring(a, b)
        # Simply use match/mismatch/gap values
        else:
            return self._default_scoring(a, b)
        
    def __str__(self):
        if self.custom_scoring is not None:
            # Use pandas DataFrame __str__ representation
            return str(self.custom_scoring)
        return f'Match: {self.match}, Mismatch: {self.mismatch}, Gap: {self.gap}'
    
 

class SequencesAnalyzer:

    # Useful for visualization
    traceback_symbols = {
        0: '↖',
        1: '↑',
        2: '←',
        3: '•'
    }

    def __init__(self, seq_a: str, seq_b: str, load_csv: bool = False) -> None:
        self.seq_a = seq_a
        self.seq_b = seq_b

        self.scoring_sys = ScoringSystems(match=2, mismatch=-1, gap=-2)
        self.edit_cost_sys = ScoringSystems(match=0, mismatch=1, gap=1)

        if load_csv:
            self.scoring_sys.load_csv('scores.csv')
            self.edit_cost_sys.load_csv('edit_cost.csv')


    def local_alignment(self) -> Tuple[str, str]:
        result: Dict[str, Any] = self.smith_waterman_algorithm()
        alignment_a, alignment_b = self._traceback(
            result_matrix=result['result_matrix'],
            traceback_matrix=result['traceback_matrix'],
            start_pos=result['score_pos'],
            global_alignment=False)
        # Calculate the maximum possible score
        max_possible_score = min(len(self.seq_a), len(self.seq_b)) * self.scoring_sys.match

        # Get the actual alignment score
        alignment_score = result['score']

        # Calculate the percentage score
        score_percentage = (alignment_score / max_possible_score) * 100

        # Update the result dictionary with the percentage score
        result['score'] = score_percentage

        print(
            f"[Local Alignment Percentage] Score={result['score']}\n"
            #f"Result:\n {result['result_matrix']}\n"
            #f"Traceback:\n {result['traceback_matrix']}\n"
            #f"Alignment:\n {alignment_a}\n {alignment_b}\n"
        )
        return alignment_a, alignment_b , score_percentage

    def similarity(self) -> int:
        result = self.needleman_wunsch_algorithm(minimize=False)
        
        print(
            f"[Similarity] Score={result['score']}\n"
            #f"{result['result_matrix']}\n"
            #f"{result['traceback_matrix']}\n"
        )
        return result['score']
    

    

    def edit_distance(self) -> int:
        result = self.needleman_wunsch_algorithm(minimize=True)

        print(
            #f"[Edit distance] Cost={result['score']}\n"
            #f"{result['result_matrix']}\n"
            #f"{result['traceback_matrix']}\n"
        )
        return result['score']

    def needleman_wunsch_algorithm(self, minimize: bool = False, alignment_cal: bool = False) -> Dict[str, Any]:
        '''
        `minimize` - set to True when calculating edit distance
        `alignment_cal` - set to True when calculating global alignment
        '''
        # 1. Prepare dimensions (required additional 1 column and 1 row)
        rows, cols = len(self.seq_a) + 1, len(self.seq_b) + 1

        # 2. Initialize matrices
        # Use grid/matrix as graph-like acyclic digraph (array cells are vertices)
        H = np.zeros(shape=(rows, cols), dtype=int)
        traceback = np.zeros(shape=(rows, cols), dtype=np.dtype('U5'))

        # 3. msadkjnsadkjn;sdf
        gapH = np.zeros(shape=(rows, cols), dtype=int)

        if minimize:
            # Edit cost calculation
            score_func = self.edit_cost_sys.score
        else:
            # Similarity calculation
            score_func = self.scoring_sys.score

        if alignment_cal:
            # Global alignment calculation -> 1st row and column need to have negative values
            sign = self.scoring_sys.gap
        else:
            # Similarity or edit cost calculation -> 1st first row and column values need to be positive
            sign = 1

        # Put sequences' letters into 1st row and 1st column (for better visualization)
        traceback[0, 1:] = np.array(list(self.seq_b), dtype=str)
        traceback[1:, 0] = np.array(list(self.seq_a), dtype=str)
    
        # 3. Top row and leftmost column, like: 0, 1, 2, 3, etc.
        H[0, :] = np.arange(start=0, stop=sign*cols, step=sign)
        H[:, 0] = np.arange(start=0, stop=sign*rows, step=sign)

        for row in range(1, rows):
            for col in range(1, cols):
                # Current pair of letters from sequence A and B
                a = self.seq_a[row - 1]
                b = self.seq_b[col - 1]

                leave_or_replace_letter = H[row -
                    1, col - 1] + score_func(a, b)

                if gapH[row - 1, col] == 0:
                    score = score_func('-', b)
                else:
                    score = 0
                delete_indel = H[row - 1, col] + score

                if gapH[row, col - 1] == 0:
                    score = score_func(a, '-')
                else:
                    score = 0
                insert_indel = H[row, col - 1] + score

                scores = [leave_or_replace_letter, delete_indel, insert_indel]

                if minimize:
                    best_action = np.argmin(scores)
                else:
                    best_action = np.argmax(scores)
                    if best_action in [1, 2]:
                        gapH[row, col] = True

                H[row, col] = scores[best_action]
                traceback[row, col] = self.traceback_symbols[best_action]
        
        #print(gapH)
        return {
            'result_matrix': H,
            'traceback_matrix': traceback,
            'score': H[-1, -1],                 # Always right-bottom corner
            'score_pos': (rows - 1, cols - 1)   # as above...
        }


    def smith_waterman_algorithm(self) -> Dict[str, Any]:
        '''
        Note: Smith-Waterman and Needleman-Wunsch algorithms
        are very similar, but because there are small differences,
        they are meant to be separated.
        '''
        # 1. Prepare dimensions (required additional 1 column and 1 row)
        rows, cols = len(self.seq_a) + 1, len(self.seq_b) + 1

        # 2. Initialize matrices
        # Use grid/matrix as graph-like acyclic digraph (array cells are vertices)
        H = np.zeros(shape=(rows, cols), dtype=int)
        traceback = np.zeros(shape=(rows, cols), dtype=np.dtype('U5'))

        # Difference 1: 1st row and 1st column are already zeroed

        # Put sequences' letters into first row and first column (better visualization)
        traceback[0, 1:] = np.array(list(self.seq_b), dtype=str)
        traceback[1:, 0] = np.array(list(self.seq_a), dtype=str)

        # 3. Top row and leftmost colum are already 0
        for row in range(1, rows):
            for col in range(1, cols):
                # Alias: current pair of letters
                a = self.seq_a[row - 1]
                b = self.seq_b[col - 1]

                score_func = self.scoring_sys.score
                leave_or_replace_letter = H[row -
                    1, col - 1] + score_func(a, b)
                delete_indel = H[row - 1, col] + score_func('-', b)
                insert_indel = H[row, col - 1] + score_func(a, '-')

                # Difference 2: That additional 0 is required (ignore negative values)
                scores = [leave_or_replace_letter,
                    delete_indel, insert_indel, 0]
                best_action = np.argmax(scores)

                H[row, col] = scores[best_action]
                traceback[row, col] = self.traceback_symbols[best_action]

        return {
            'result_matrix': H,
            'traceback_matrix': traceback,
            'score': H.max(),
            # Force numpy to return last result
            # Source: (Step 2: Backtracing) https://tiefenauer.github.io/blog/smith-waterman
            'score_pos': np.unravel_index(np.argmax(H, axis=None), H.shape)
        }

    def _traceback(self, result_matrix, traceback_matrix, start_pos: Tuple[int, int], global_alignment: bool) -> Tuple[str, str]:
        seq_a_aligned = ''
        seq_b_aligned = ''

        # 1. Select starting point
        row, col = start_pos

        if global_alignment:
            # Terminate when top left corner (0,0) is reached (end of path)
            end_condition_reached = lambda row, col: row == 0 and col == 0
        else:
            # Terminate when 0 is reached
            end_condition_reached = lambda row, col: result_matrix[row, col] == 0

        while not end_condition_reached(row, col):
            symbol = traceback_matrix[row, col]
            if row == 0:
                symbol = '←'
            if col == 0:
                symbol = '↑'
            # Use arrows to navigate and collect letters (in reversed order)
            # Shift/reverse indexes by one beforehand (we want to get the letter that arrow points to)
            if symbol == '↖':
                row -= 1
                col -= 1
                letter_a, letter_b = self.seq_a[row], self.seq_b[col]
            elif symbol == '↑':
                row -= 1
                letter_a, letter_b = self.seq_a[row], '-'
            elif symbol == '←':
                col -= 1
                letter_a, letter_b = '-', self.seq_b[col]

            # Acumulate letter (in reversed order)
            seq_a_aligned += letter_a
            seq_b_aligned += letter_b

        # Reverse strings (traceback goes from bottom-right to top-left)
        return seq_a_aligned[::-1], seq_b_aligned[::-1]

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <sequence_a_file> <sequence_b_file> [--summary as --S] [--similarity as --s] [--edit-distance as --e ] [--alignment <global/local>]")
        return

    sequence_a_file = sys.argv[1]
    sequence_b_file = sys.argv[2]
    #summary = '-S' in sys.argv
    summary = False
    similarity = True
    #similarity = '-s' in sys.argv
    #edit_distance = '-e' in sys.argv
    # alignment = None
    # if '--alignment' in sys.argv:
    #     alignment_index = sys.argv.index('--alignment')
    #     alignment = sys.argv[alignment_index + 1] if alignment_index + 1 < len(sys.argv) else None

    # Load CSV files
    scoring_sys = ScoringSystems()
    scoring_sys.load_csv(SCORES_CSV)
    edit_cost_sys = ScoringSystems()
    edit_cost_sys.load_csv(EDIT_COST_CSV)

    with open(sequence_a_file, 'r') as file_a, open(sequence_b_file, 'r') as file_b:
        sequence_a = ''.join(line.strip()[:500] for line in file_a if not line.startswith('>'))
        sequence_b = ''.join(line.strip()[:500] for line in file_b if not line.startswith('>'))

    analyzer = SequencesAnalyzer(sequence_a, sequence_b )

    if summary:
        pass 
    elif similarity:
        #analyzer.similarity()
        alignment_a, alignment_b = analyzer.local_alignment()
        if analyzer.similarity() == 1000:
            print("DNA matched")
        else:
            print("Not matched")

def process_uploaded_file(file):
    data = b''  # Initialize as bytes
    total_length = 0
    for line in file:
        line = line.strip()
        if line.startswith(b'>'):  # Compare with byte string
            continue  # Skip header lines
        remaining_length = 500 - total_length
        if remaining_length <= 0:
            break  # Exit loop if the limit is reached
        # Concatenate the line directly
        data += line[:remaining_length]
        total_length += len(data)
    return data.decode()  # Decode the result back to string


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'file_a' not in request.files or 'file_b' not in request.files:
        return "Please upload both files."

    file_a = request.files['file_a']
    file_b = request.files['file_b']

    
    sequence_a = process_uploaded_file(file_a)
    sequence_b = process_uploaded_file(file_b)

    similarity = True

    # Load CSV files
    scoring_sys = ScoringSystems()
    scoring_sys.load_csv(SCORES_CSV)
    edit_cost_sys = ScoringSystems()
    edit_cost_sys.load_csv(EDIT_COST_CSV)

    analyzer = SequencesAnalyzer(sequence_a, sequence_b)

    if similarity:
       # alignment_a, alignment_b = analyzer.local_alignment()
        similarity_score = int(analyzer.similarity())  # Convert to integer
        match_status = "DNA MATCH" if similarity_score == 1000 else "DNA Not MATCH"
        similarity_percentage = int((similarity_score / 1000) * 100)  # Convert to integer
    else:
        pass
    return jsonify({
        #"alignment_a": alignment_a,
        #"alignment_b": alignment_b,
        "similarity_percentage": similarity_percentage,
        #"similarity_score": similarity_score,
        "match_status": match_status
    })
if __name__ == "__main__":
    app.run(debug=True)